// Copyright 2023 The Flutter team. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as image_lib;
import 'package:live_object_detection_ssd_mobilenet/models/recognition.dart';
import 'package:live_object_detection_ssd_mobilenet/utils/image_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

enum Codes {
  init,
  busy,
  ready,
  detect,
  result,
}

class Command {
  const Command(this.code, {this.args});

  final Codes code;
  final List<Object>? args;
}

class Detector {
  static const String modelPath = 'assets/models/ssd_mobilenet.tflite';
  static const String labelPath = 'assets/models/ssd_mobilenet.txt';
  // static const String modelPath = 'assets/models/yolov5n_float32.tflite';
  // static const String labelPath = 'assets/models/yolov5n_float32.txt';

  Detector(
    this.isolate,
    this.interpreter,
    this.labels,
  );

  final      Isolate      isolate;
  late final Interpreter  interpreter;
  late final List<String> labels;

  late final SendPort sendPort;
  
  bool isReady = false;

  final StreamController<Map<String,dynamic>> resultsStream = StreamController<Map<String,dynamic>>();

  static Future<Detector> start() async {
    
    final ReceivePort receivePort = ReceivePort();
    
    final Isolate isolate = await Isolate.spawn(
      DetectorServer.run,
      receivePort.sendPort,
    );

    final Detector result = Detector(
      isolate,
      await loadModel(),
      await loadLabels(),
    );

    receivePort.listen((message) {
      result.handleCommand(message as Command);
    });
    return result;
  }

  static Future<Interpreter> loadModel() async {

    final interpreterOptions = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      interpreterOptions.addDelegate(XNNPackDelegate());
    }
    if (Platform.isIOS) {
      final gpuDelegate = GpuDelegate(
        options: GpuDelegateOptions(
          allowPrecisionLoss: true,
        ),
      );
      interpreterOptions.addDelegate(gpuDelegate);
    }

    return Interpreter.fromAsset(
      modelPath,
      options: interpreterOptions..threads = 4,
    );
  }

  static Future<List<String>> loadLabels() async {
    return (await rootBundle.loadString(labelPath)).split('\n');
  }

  /// Starts CameraImage processing
  void processFrame(CameraImage cameraImage) {
    if (isReady) {
      sendPort.send(
        Command(
          Codes.detect,
          args: [cameraImage],
        ),
      );
    }
  }

  /// Handler invoked when a message is received from the port communicating
  /// with the database server.
  void handleCommand(Command command) {
    switch (command.code) {
      case Codes.init:
        sendPort = command.args?[0] as SendPort;
        // ----------------------------------------------------------------------
        // Before using platform channels and plugins from background isolates we
        // need to register it with its root isolate. This is achieved by
        // acquiring a [RootIsolateToken] which the background isolate uses to
        // invoke [BackgroundIsolateBinaryMessenger.ensureInitialized].
        // ----------------------------------------------------------------------
        RootIsolateToken rootIsolateToken = RootIsolateToken.instance!;
        sendPort.send(Command(Codes.init, args: [
          rootIsolateToken,
          interpreter.address,
          labels,
        ]));
      case Codes.ready:
        isReady = true;
      case Codes.busy:
        isReady = false;
      case Codes.result:
        isReady = true;
        resultsStream.add(command.args?[0] as Map<String, dynamic>);
      default:
        debugPrint('Detector unrecognized command: ${command.code}');
    }
  }

  /// Kills the background isolate and its detector server.
  void stop() {
    isolate.kill();
  }
}

/// The portion of the [Detector] that runs on the background isolate.
///
/// This is where we use the new feature Background Isolate Channels, which
/// allows us to use plugins from background isolates.
class DetectorServer {
  /// Input size of image (height = width = 300)
  static const int mlModelInputSize = 300;
  // static const int mlModelInputSize = 640;

  /// Result confidence threshold
  static const double confidence = 0.5;
  Interpreter? interpreter;
  List<String>? labels;

  DetectorServer(this.sendPort);

  final SendPort sendPort;

  // ----------------------------------------------------------------------
  // Here the plugin is used from the background isolate.
  // ----------------------------------------------------------------------

  /// The main entrypoint for the background isolate sent to [Isolate.spawn].
  static void run(SendPort sendPort) {
    ReceivePort receivePort = ReceivePort();
    final DetectorServer server = DetectorServer(sendPort);
    receivePort.listen((message) async {
      final Command command = message as Command;
      await server._handleCommand(command);
    });
    // receivePort.sendPort - used by UI isolate to send commands to the service receiverPort
    sendPort.send(Command(Codes.init, args: [receivePort.sendPort]));
  }

  /// Handle the [command] received from the [ReceivePort].
  Future<void> _handleCommand(Command command) async {
    switch (command.code) {
      case Codes.init:
        // ----------------------------------------------------------------------
        // The [RootIsolateToken] is required for
        // [BackgroundIsolateBinaryMessenger.ensureInitialized] and must be
        // obtained on the root isolate and passed into the background isolate via
        // a [SendPort].
        // ----------------------------------------------------------------------
        RootIsolateToken rootIsolateToken =
            command.args?[0] as RootIsolateToken;
        // ----------------------------------------------------------------------
        // [BackgroundIsolateBinaryMessenger.ensureInitialized] for each
        // background isolate that will use plugins. This sets up the
        // [BinaryMessenger] that the Platform Channels will communicate with on
        // the background isolate.
        // ----------------------------------------------------------------------
        BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);
        interpreter = Interpreter.fromAddress(command.args?[1] as int);
        labels = command.args?[2] as List<String>;
        sendPort.send(const Command(Codes.ready));
      case Codes.detect:
        sendPort.send(const Command(Codes.busy));
        _convertCameraImage(command.args?[0] as CameraImage);
      default:
        debugPrint('_DetectorService unrecognized command ${command.code}');
    }
  }

  void _convertCameraImage(CameraImage cameraImage) {
    var preConversionTime = DateTime.now().millisecondsSinceEpoch;

    convertCameraImageToImage(cameraImage).then(
      (image) {
        if (image != null) {
          if (Platform.isAndroid) {
            image = image_lib.copyRotate(image, angle: 90);
          }
          final results = analyseImage(image, preConversionTime);
          sendPort.send(Command(Codes.result, args: [results]));
        }
      },
    );
  }

  Map<String, dynamic> analyseImage(
    image_lib.Image image,
    int preConversionTime
  ) {
    var conversionElapsedTime = DateTime.now().millisecondsSinceEpoch - preConversionTime;

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    /// Pre-process the image
    /// Resizing image for model [300, 300]
    final image_lib.Image imageInput = image_lib.copyResize(
      image,
      width:  mlModelInputSize,
      height: mlModelInputSize,
    );

    // Creating matrix representation, [300, 300, 3]
    final imageMatrix = List.generate(
      imageInput.height,
      (y) => List.generate(
        imageInput.width,
        (x) {
          final pixel = imageInput.getPixel(x, y);
          return [pixel.r, pixel.g, pixel.b];
        },
      ),
    );

    var preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;

    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    final output = runInference(imageMatrix);

    // Location
    final locationsRaw = output.first.first as List<List<double>>;

    final List<Rect> locations = locationsRaw
        .map((list) => list.map((value) => (value * mlModelInputSize)).toList())
        .map((rect) => Rect.fromLTRB(rect[1], rect[0], rect[3], rect[2]))
        .toList();

    // Classes
    final classesRaw = output.elementAt(1).first as List<double>;
    final classes = classesRaw.map((value) => value.toInt()).toList();

    // Scores
    final scores = output.elementAt(2).first as List<double>;

    // Number of detections
    final numberOfDetectionsRaw = output.last.first as double;
    final numberOfDetections = numberOfDetectionsRaw.toInt();

    final List<String> classification = [];
    for (var i = 0; i < numberOfDetections; i++) {
      classification.add(labels![classes[i]]);
    }

    /// Generate recognitions
    List<Recognition> recognitions = [];
    for (int i = 0; i < numberOfDetections; i++) {
      // Prediction score
      var score = scores[i];
      // Label string
      var label = classification[i];

      if (score > confidence) {
        recognitions.add(
          Recognition(i, label, score, locations[i]),
        );
      }
    }

    var inferenceElapsedTime =
        DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;

    var totalElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preConversionTime;

    return {
      "recognitions": recognitions,
      "stats": <String, String>{
        'Conversion time:': conversionElapsedTime.toString(),
        'Pre-processing time:': preProcessElapsedTime.toString(),
        'Inference time:': inferenceElapsedTime.toString(),
        'Total prediction time:': totalElapsedTime.toString(),
        'Frame': '${image.width} X ${image.height}',
      },
    };
  }

  /// Object detection main function
  List<List<Object>> runInference(
    List<List<List<num>>> imageMatrix,
  ){
    // Set input tensor [1, 300, 300, 3]
    final input = [imageMatrix];

    // Set output tensor
    // Locations: [1, 10, 4]
    // Classes: [1, 10],
    // Scores: [1, 10],
    // Number of detections: [1]
    final output = {
      0: [List<List<num>>.filled(10, List<num>.filled(4, 0))],
      1: [List<num>.filled(10, 0)],
      2: [List<num>.filled(10, 0)],
      3: [0.0],
    };

    interpreter!.runForMultipleInputs([input], output);
    return output.values.toList();
  }
}
