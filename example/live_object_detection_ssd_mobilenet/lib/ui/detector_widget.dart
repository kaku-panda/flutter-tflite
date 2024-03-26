import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:live_object_detection_ssd_mobilenet/models/recognition.dart';
import 'package:live_object_detection_ssd_mobilenet/models/screen_params.dart';
import 'package:live_object_detection_ssd_mobilenet/service/detector_service.dart';
import 'package:live_object_detection_ssd_mobilenet/ui/box_widget.dart';
import 'package:live_object_detection_ssd_mobilenet/ui/stats_widget.dart';

/// sends each frame for inference
class DetectorWidget extends StatefulWidget {
  
  const DetectorWidget({super.key});

  @override
  State<DetectorWidget> createState() => DetectorWidgetState();
}

class DetectorWidgetState extends State<DetectorWidget> with WidgetsBindingObserver {
  
  late List<CameraDescription> cameras;

  CameraController? cameraController;
  get controller => cameraController;

  Detector? detector;
  StreamSubscription? subscription;

  List<Recognition>? results;

  Map<String, String>? stats;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    initStateAsync();
  }

  void initStateAsync() async {
    // initialize preview and CameraImage stream
    initializeCamera();
    // Spawn a new isolate
    Detector.start().then(
      (instance) {
        setState(() {
          detector = instance;
          subscription = instance.resultsStream.stream.listen(
            (values) {
              setState(() {
                results = values['recognitions'];
                stats   = values['stats'];
              });
            },
          );
        });
      },
    );
  }

  void initializeCamera() async {
    cameras = await availableCameras();
    cameraController = CameraController(
      cameras[0],
      ResolutionPreset.low,
      enableAudio: false,
    )..initialize().then((_) async {
      await controller.startImageStream(onLatestImageAvailable);
      setState(() {});
      ScreenParams.previewSize = controller.value.previewSize!;
    });
  }

  @override
  Widget build(BuildContext context) {
    
    if (cameraController == null || !controller.value.isInitialized) {
      return const SizedBox.shrink();
    }

    var aspect = 1 / controller.value.aspectRatio;

    return Stack(
      children: [
        AspectRatio(
          aspectRatio: aspect,
          child: CameraPreview(controller),
        ),
        statsWidget(),
        AspectRatio(
          aspectRatio: aspect,
          child: boundingBoxes(),
        ),
      ],
    );
  }

  Widget statsWidget() => (stats != null)
      ? Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            color: Colors.white.withAlpha(150),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: stats!.entries
                    .map((e) => StatsWidget(e.key, e.value))
                    .toList(),
              ),
            ),
          ),
        )
      : const SizedBox.shrink();

  /// Returns Stack of bounding boxes
  Widget boundingBoxes() {
    if (results == null) {
      return const SizedBox.shrink();
    }
    return Stack(
        children: results!.map((box) => BoxWidget(result: box)).toList());
  }

  /// Callback to receive each frame [CameraImage] perform inference on it
  void onLatestImageAvailable(CameraImage cameraImage) async {
    detector?.processFrame(cameraImage);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async {
    switch (state) {
      case AppLifecycleState.inactive:
        cameraController?.stopImageStream();
        detector?.stop();
        subscription?.cancel();
        break;
      case AppLifecycleState.resumed:
        initStateAsync();
        break;
      default:
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    cameraController?.dispose();
    detector?.stop();
    subscription?.cancel();
    super.dispose();
  }
}
