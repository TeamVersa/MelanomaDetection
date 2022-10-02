// Copyright 2013 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';
import 'dart:io';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:syncfusion_flutter_gauges/gauges.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool? isFemale = false;
  final bodyLocations = [
    'unknown',
    'head/neck',
    'upper extremity',
    'lower extremity',
    'torso',
    'palms/soles',
    'oral/genital'
  ];
  String? bodyLocation = 'unknown';
  int? age = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(slivers: [
        SliverAppBar(
          flexibleSpace: FlexibleSpaceBar(
            background: Padding(
              padding: const EdgeInsets.all(10.0),
              child: ImageIcon(
                AssetImage('assets/melascan.png'),
                color: Colors.white,
              ),
              // Text(
              //     '\n\nEarly-stage detection of melanoma is crucial.\nWe provide decision support for doctors to rapidly detect it using AI.\n\n1. Enter patient data below.\n2. Take picture of mole in the center.\n3. If resulting melanoma probability is high, please consider referring the patient to a dermatologist.\n\nThe result only analyses melanoma, other skin cancer types will not be recognized.',
              //     style:
              //         TextStyle(fontSize: 25, fontWeight: FontWeight.w300)),
            ),
            collapseMode: CollapseMode.pin,
            titlePadding: EdgeInsets.all(5),
            title: Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                ImageIcon(
                  AssetImage('assets/melascan.png'),
                  size: 60,
                  color: Colors.white,
                ),
                SizedBox(width: 5),
                Text('Mela',
                    style:
                        TextStyle(fontSize: 40, fontWeight: FontWeight.w300)),
                Text('Scan',
                    style:
                        TextStyle(fontSize: 40, fontWeight: FontWeight.w100)),
              ],
            ),
          ),
          pinned: true,
          expandedHeight: MediaQuery.of(context).size.height * 2 / 3,
          backgroundColor: Theme.of(context).primaryColor,
        ),
        SliverList(
          delegate: SliverChildListDelegate(
            [
              SizedBox(
                  height: MediaQuery.of(context).size.height * 1 / 3,
                  child: Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(
                            "DISCLAIMER: This app should be used only as an assistance tool to help distinguish melanoma. For final diagnosis, please refer to the biopsy results.",
                            style: TextStyle(color: Colors.grey)),
                      ),
                      Spacer(flex: 1),
                      Text(
                        'scroll down to start',
                        style: TextStyle(
                          fontSize: 30,
                          color: Theme.of(context).primaryColor,
                        ),
                      ),
                      Icon(
                        Icons.expand_more,
                        size: 60,
                        color: Theme.of(context).primaryColor,
                      ),
                      Spacer(flex: 3),
                    ],
                  )),
              SizedBox(
                height: MediaQuery.of(context).size.height -
                    AppBar().preferredSize.height -
                    MediaQuery.of(context).padding.top,
                child: Column(
                  children: [
                    ListTile(
                        title: Text(
                      'Patient Information:',
                      style: TextStyle(
                        fontSize: 24,
                        // fontWeight: FontWeight.bold,
                      ),
                    )),
                    Divider(),
                    ListTile(
                        title: Text('Sex:'),
                        trailing: SizedBox(
                            width: 200,
                            child: RadioListTile(
                              title: Text('Male'),
                              secondary: Icon(Icons.male),
                              value: false,
                              groupValue: isFemale,
                              activeColor: Theme.of(context).primaryColor,
                              onChanged: (value) {
                                setState(() {
                                  isFemale = value;
                                });
                              },
                            ))),
                    ListTile(
                      trailing: SizedBox(
                        width: 200,
                        child: RadioListTile(
                          title: Text('Female'),
                          secondary: Icon(Icons.female),
                          value: true,
                          groupValue: isFemale,
                          activeColor: Theme.of(context).primaryColor,
                          onChanged: (value) {
                            setState(() {
                              isFemale = value;
                            });
                          },
                        ),
                      ),
                    ),
                    //Divider(),
                    ListTile(
                        title: Text('Age:'),
                        trailing: SizedBox(
                          width: 50,
                          child: TextFormField(
                            initialValue: '0',
                            onChanged: (a) {
                              if (a.isNotEmpty) age = int.parse(a);
                            },
                            keyboardType: TextInputType.number,
                            inputFormatters: <TextInputFormatter>[
                              FilteringTextInputFormatter.digitsOnly
                            ],
                            // cursorColor: Theme.of(context).primaryColor,
                          ),
                        )),
                    //Divider(),
                    ListTile(
                        title: Text('Mole on body part:'),
                        trailing: DropdownButton<String>(
                            value: bodyLocation,
                            items: bodyLocations
                                .map((val) => DropdownMenuItem(
                                    child: Text(val), value: val))
                                .toList(),
                            onChanged: (value) {
                              setState(() {
                                bodyLocation = value;
                              });
                            })),
                  ],
                ),
              ),
            ],
          ),
        ),
      ]),
      floatingActionButton: ElevatedButton(
        child: Icon(Icons.camera_enhance, size: 40),
        onPressed: () {
          Navigator.of(context).push(MaterialPageRoute(builder: (context) {
            return CameraExampleHome();
          }));
        },
        style: ElevatedButton.styleFrom(
          shape: CircleBorder(),
          padding: EdgeInsets.all(20),
          backgroundColor: Theme.of(context).primaryColor, // <-- Button color
          foregroundColor: Colors.white, // <-- Splash color
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

class SubmitPage extends StatefulWidget {
  final XFile image;
  final String serverAddr;

  const SubmitPage({super.key, required this.image, required this.serverAddr});

  @override
  State<SubmitPage> createState() => _SubmitPageState();
}

class _SubmitPageState extends State<SubmitPage> {
  late Future<Map<String, dynamic>> result;

  void showInSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(message)));
    }
  }

  Future<Map<String, dynamic>> upload(String url, XFile imageFile) async {
    try {
      var stream = http.ByteStream(imageFile.openRead());
      stream.cast();
      var length = await imageFile.length();

      var uri = Uri.parse(url);

      var request = http.MultipartRequest("POST", uri);
      var multipartFile =
          http.MultipartFile('image', stream, length, filename: imageFile.path);

      request.files.add(multipartFile);
      var response = await request.send();
      print(response.statusCode);
      var responseData = await response.stream.toBytes();
      var responseString = String.fromCharCodes(responseData);
      print(responseString);

      return jsonDecode(responseString);
    } on Exception catch (e) {
      showInSnackBar('$e');
      return {};
    }
  }

  @override
  void initState() {
    super.initState();
    result = upload('http://${widget.serverAddr}/melanoma', widget.image);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Analysis Result'),
        titleTextStyle: TextStyle(
          fontSize: 32,
          color: Colors.white,
        ),
        centerTitle: true,
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            SizedBox(
              width: MediaQuery.of(context).size.width,
              height: MediaQuery.of(context).size.width,
              child: Image.file(
                File(widget.image.path),
                fit: BoxFit.fitWidth,
              ),
            ),
            Spacer(),
            FutureBuilder(
                future: result,
                builder: ((context, snapshot) {
                  if (snapshot.hasData) {
                    var result = snapshot.data!;
                    return Column(
                      children: [
                        // ListTile(
                        //     title: Text('Prediction:'),
                        //     trailing: Text('${result['prediction']}')),
                        ListTile(
                            title: Text('Probability of Melanoma:'),
                            trailing: Text(
                                '${(result['confidence'] * 10000).round() / 100}%')),
                        ListTile(
                            title: SfLinearGauge(
                          ranges: [
                            LinearGaugeRange(
                                startValue: 0,
                                endValue: 1 / 3 * 100,
                                color: Colors.green),
                            LinearGaugeRange(
                                startValue: 1 / 3 * 100,
                                endValue: 2 / 3 * 100,
                                color: Colors.orange),
                            LinearGaugeRange(
                                startValue: 2 / 3 * 100,
                                endValue: 100,
                                color: Colors.red)
                          ],
                          markerPointers: [
                            LinearShapePointer(
                              value: result['confidence'] * 100,
                            ),
                          ],
                        ))
                      ],
                    );
                  } else if (snapshot.hasError) {
                    return Text('${snapshot.error}');
                  } else {
                    return Column(
                      children: [
                        Text(
                          'Analysing...',
                          style: TextStyle(
                              fontSize: 20,
                              color: Theme.of(context).primaryColor),
                        ),
                        SizedBox(height: 20),
                        CircularProgressIndicator(
                            color: Theme.of(context).primaryColor),
                      ],
                    );
                  }
                })),
            Spacer(),
          ],
        ),
      ),
    );
  }
}

/// Camera example home widget.
class CameraExampleHome extends StatefulWidget {
  /// Default Constructor
  const CameraExampleHome({Key? key}) : super(key: key);

  @override
  State<CameraExampleHome> createState() {
    return _CameraExampleHomeState();
  }
}

void _logError(String code, String? message) {
  if (message != null) {
    print('Error: $code\nError Message: $message');
  } else {
    print('Error: $code');
  }
}

class _CameraExampleHomeState extends State<CameraExampleHome>
    with WidgetsBindingObserver, TickerProviderStateMixin {
  CameraController? controller;
  XFile? imageFile;
  double _minAvailableExposureOffset = 0.0;
  double _maxAvailableExposureOffset = 0.0;
  double _currentExposureOffset = 0.0;
  late AnimationController _exposureModeControlRowAnimationController;
  late Animation<double> _exposureModeControlRowAnimation;
  late AnimationController _focusModeControlRowAnimationController;
  late Animation<double> _focusModeControlRowAnimation;
  double _minAvailableZoom = 1.0;
  double _maxAvailableZoom = 1.0;
  double _currentScale = 1.0;
  double _baseScale = 1.0;

  String serverAddr = '10.183.53.222:5000';

  // Counting pointers (number of user fingers on screen)
  int _pointers = 0;

  @override
  void initState() {
    super.initState();
    _ambiguate(WidgetsBinding.instance)?.addObserver(this);

    _exposureModeControlRowAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _exposureModeControlRowAnimation = CurvedAnimation(
      parent: _exposureModeControlRowAnimationController,
      curve: Curves.easeInCubic,
    );
    _focusModeControlRowAnimationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _focusModeControlRowAnimation = CurvedAnimation(
      parent: _focusModeControlRowAnimationController,
      curve: Curves.easeInCubic,
    );

    //default to main camera
    if (_cameras.isNotEmpty) {
      onNewCameraSelected(_cameras[0])
          .then((value) => setFlashMode(FlashMode.torch));
    }
  }

  @override
  void dispose() {
    setFlashMode(FlashMode.off);
    _ambiguate(WidgetsBinding.instance)?.removeObserver(this);
    _exposureModeControlRowAnimationController.dispose();
    super.dispose();
  }

  // #docregion AppLifecycle
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = controller;

    // App state changed before we got the chance to initialize.
    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      onNewCameraSelected(cameraController.description);
    }
  }
  // #enddocregion AppLifecycle

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: const Text('Scan the Mole.'),
        titleTextStyle: TextStyle(
            fontSize: 32,
            color: Colors.white,
            shadows: <Shadow>[Shadow(color: Colors.black45, blurRadius: 15.0)]),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.transparent,
        actions: [
          IconButton(
              onPressed: () {
                Navigator.of(context)
                    .push(MaterialPageRoute(builder: (context) {
                  return Scaffold(
                      appBar: AppBar(title: Text('Settings')),
                      body: Column(
                        children: [
                          Text('Server Address: $serverAddr'),
                          TextField(
                            onSubmitted: (value) {
                              serverAddr = value;
                            },
                          ),
                        ],
                      ));
                }));
              },
              icon: Icon(Icons.settings, color: Colors.white, shadows: <Shadow>[
                Shadow(color: Colors.black45, blurRadius: 15.0)
              ]))
        ],
      ),
      body: Stack(
        children: [
          _cameraPreviewWidget(),
          Column(
            children: <Widget>[
              SizedBox(
                height: MediaQuery.of(context).padding.top + kToolbarHeight,
              ),
              _modeControlRowWidget(),
              Expanded(
                child: Container(
                    // decoration: BoxDecoration(
                    //   color: Colors.black,
                    //   border: Border.all(
                    //     color: controller != null &&
                    //             controller!.value.isRecordingVideo
                    //         ? Colors.redAccent
                    //         : Colors.grey,
                    //     width: 3.0,
                    //   ),
                    // ),
                    // child: Padding(
                    //   padding: const EdgeInsets.all(1.0),
                    //   child: Center(
                    //     child: _cameraPreviewWidget(),
                    //   ),
                    // ),
                    ),
              ),
              _captureControlRowWidget(),
              const SizedBox(height: 50),
            ],
          ),
        ],
      ),
    );
  }

  /// Display the preview from the camera (or a message if the preview is not available).
  Widget _cameraPreviewWidget() {
    final CameraController? cameraController = controller;

    if (cameraController == null || !cameraController.value.isInitialized) {
      return const Text('Select a camera');
    } else {
      final mediaSize = MediaQuery.of(context).size;
      final scale = 1 / (controller!.value.aspectRatio * mediaSize.aspectRatio);

      return Listener(
          onPointerDown: (_) => _pointers++,
          onPointerUp: (_) => _pointers--,
          child: ClipRect(
            clipper: _MediaSizeClipper(mediaSize),
            child: Transform.scale(
              scale: scale,
              alignment: Alignment.topCenter,
              child: CameraPreview(
                controller!,
                child: LayoutBuilder(builder:
                    (BuildContext context, BoxConstraints constraints) {
                  return GestureDetector(
                    behavior: HitTestBehavior.opaque,
                    onScaleStart: _handleScaleStart,
                    onScaleUpdate: _handleScaleUpdate,
                    onTapDown: (TapDownDetails details) =>
                        onViewFinderTap(details, constraints),
                  );
                }),
              ),
            ),
          ));
    }
  }

  void _handleScaleStart(ScaleStartDetails details) {
    _baseScale = _currentScale;
  }

  Future<void> _handleScaleUpdate(ScaleUpdateDetails details) async {
    // When there are not exactly two fingers on screen don't scale
    if (controller == null || _pointers != 2) {
      return;
    }

    _currentScale = (_baseScale * details.scale)
        .clamp(_minAvailableZoom, _maxAvailableZoom);

    await controller!.setZoomLevel(_currentScale);
  }

  /// Display a bar with buttons to change the flash and exposure modes
  Widget _modeControlRowWidget() {
    return Column(
      children: <Widget>[
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: <Widget>[
            IconButton(
              icon: const Icon(Icons.highlight, shadows: <Shadow>[
                Shadow(color: Colors.black45, blurRadius: 15.0)
              ]),
              color: controller?.value.flashMode == FlashMode.torch
                  ? Theme.of(context).primaryColor
                  : Colors.white,
              onPressed: controller != null
                  ? () => onSetFlashModeButtonPressed(
                      controller?.value.flashMode == FlashMode.torch
                          ? FlashMode.off
                          : FlashMode.torch)
                  : null,
            ),
            // The exposure and focus mode are currently not supported on the web.
            ...!kIsWeb
                ? <Widget>[
                    IconButton(
                      icon: const Icon(Icons.exposure, shadows: <Shadow>[
                        Shadow(color: Colors.black45, blurRadius: 15.0)
                      ]),
                      color: Colors.white,
                      onPressed: controller != null
                          ? onExposureModeButtonPressed
                          : null,
                    ),
                    IconButton(
                      icon: const Icon(Icons.filter_center_focus,
                          shadows: <Shadow>[
                            Shadow(color: Colors.black45, blurRadius: 15.0)
                          ]),
                      color: Colors.white,
                      onPressed:
                          controller != null ? onFocusModeButtonPressed : null,
                    )
                  ]
                : <Widget>[],
            IconButton(
              icon: const Icon(Icons.flip_camera_ios, shadows: <Shadow>[
                Shadow(color: Colors.black45, blurRadius: 15.0)
              ]),
              color: Colors.white,
              onPressed: onChangeCameraButtonPressed,
            ),
          ],
        ),
        //_flashModeControlRowWidget(),
        _exposureModeControlRowWidget(),
        _focusModeControlRowWidget(),
      ],
    );
  }

  Widget _exposureModeControlRowWidget() {
    final ButtonStyle styleAuto = TextButton.styleFrom(
      // TODO(darrenaustin): Migrate to new API once it lands in stable: https://github.com/flutter/flutter/issues/105724
      // ignore: deprecated_member_use
      primary: controller?.value.exposureMode == ExposureMode.auto
          ? Theme.of(context).primaryColor
          : Colors.white,
    );
    final ButtonStyle styleLocked = TextButton.styleFrom(
      // TODO(darrenaustin): Migrate to new API once it lands in stable: https://github.com/flutter/flutter/issues/105724
      // ignore: deprecated_member_use
      primary: controller?.value.exposureMode == ExposureMode.locked
          ? Theme.of(context).primaryColor
          : Colors.white,
    );

    return SizeTransition(
      sizeFactor: _exposureModeControlRowAnimation,
      child: ClipRect(
        child: Container(
          color: const Color.fromARGB(100, 100, 100, 100),
          child: Column(
            children: <Widget>[
              const Center(
                child: Text('Exposure Mode',
                    style: TextStyle(color: Colors.white)),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: <Widget>[
                  TextButton(
                    style: styleAuto,
                    onPressed: controller != null
                        ? () =>
                            onSetExposureModeButtonPressed(ExposureMode.auto)
                        : null,
                    onLongPress: () {
                      if (controller != null) {
                        controller!.setExposurePoint(null);
                        showInSnackBar('Resetting exposure point');
                      }
                    },
                    child: const Text('AUTO'),
                  ),
                  TextButton(
                    style: styleLocked,
                    onPressed: controller != null
                        ? () =>
                            onSetExposureModeButtonPressed(ExposureMode.locked)
                        : null,
                    child: const Text('LOCKED'),
                  ),
                  TextButton(
                    style: styleLocked,
                    onPressed: controller != null
                        ? () => controller!.setExposureOffset(0.0)
                        : null,
                    child: const Text('RESET OFFSET'),
                  ),
                ],
              ),
              const Center(
                child: Text('Exposure Offset',
                    style: TextStyle(color: Colors.white)),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: <Widget>[
                  Text(_minAvailableExposureOffset.toString(),
                      style: const TextStyle(color: Colors.white)),
                  Slider(
                    value: _currentExposureOffset,
                    min: _minAvailableExposureOffset,
                    max: _maxAvailableExposureOffset,
                    label: _currentExposureOffset.toString(),
                    onChanged: _minAvailableExposureOffset ==
                            _maxAvailableExposureOffset
                        ? null
                        : setExposureOffset,
                  ),
                  Text(_maxAvailableExposureOffset.toString(),
                      style: const TextStyle(color: Colors.white)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _focusModeControlRowWidget() {
    final ButtonStyle styleAuto = TextButton.styleFrom(
      // TODO(darrenaustin): Migrate to new API once it lands in stable: https://github.com/flutter/flutter/issues/105724
      // ignore: deprecated_member_use
      primary: controller?.value.focusMode == FocusMode.auto
          ? Theme.of(context).primaryColor
          : Colors.white,
    );
    final ButtonStyle styleLocked = TextButton.styleFrom(
      // TODO(darrenaustin): Migrate to new API once it lands in stable: https://github.com/flutter/flutter/issues/105724
      // ignore: deprecated_member_use
      primary: controller?.value.focusMode == FocusMode.locked
          ? Theme.of(context).primaryColor
          : Colors.white,
    );

    return SizeTransition(
      sizeFactor: _focusModeControlRowAnimation,
      child: ClipRect(
        child: Container(
          color: const Color.fromARGB(100, 100, 100, 100),
          child: Column(
            children: <Widget>[
              const Center(
                child:
                    Text('Focus Mode', style: TextStyle(color: Colors.white)),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: <Widget>[
                  TextButton(
                    style: styleAuto,
                    onPressed: controller != null
                        ? () => onSetFocusModeButtonPressed(FocusMode.auto)
                        : null,
                    onLongPress: () {
                      if (controller != null) {
                        controller!.setFocusPoint(null);
                      }
                      showInSnackBar('Resetting focus point');
                    },
                    child: const Text('AUTO'),
                  ),
                  TextButton(
                    style: styleLocked,
                    onPressed: controller != null
                        ? () => onSetFocusModeButtonPressed(FocusMode.locked)
                        : null,
                    child: const Text('LOCKED'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Display the control bar with buttons to take pictures and record videos.
  Widget _captureControlRowWidget() {
    final CameraController? cameraController = controller;

    return Center(
      child: IconButton(
        icon: const Icon(Icons.circle_outlined,
            shadows: <Shadow>[Shadow(color: Colors.black45, blurRadius: 15.0)]),
        iconSize: 100,
        color: Colors.white,
        padding: EdgeInsets.zero,
        onPressed: cameraController != null &&
                cameraController.value.isInitialized &&
                !cameraController.value.isRecordingVideo
            ? onTakePictureButtonPressed
            : null,
      ),
    );
  }

  void onChangeCameraButtonPressed() {
    //find current camera index
    if (controller != null) {
      var index = _cameras.indexOf(controller!.description);
      onNewCameraSelected(_cameras[(index + 1) % _cameras.length]);
    }
  }

  String timestamp() => DateTime.now().millisecondsSinceEpoch.toString();

  void showInSnackBar(String message) {
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  void onViewFinderTap(TapDownDetails details, BoxConstraints constraints) {
    if (controller == null) {
      return;
    }

    final CameraController cameraController = controller!;

    final Offset offset = Offset(
      details.localPosition.dx / constraints.maxWidth,
      details.localPosition.dy / constraints.maxHeight,
    );
    cameraController.setExposurePoint(offset);
    cameraController.setFocusPoint(offset);
  }

  Future<void> onNewCameraSelected(CameraDescription cameraDescription) async {
    final CameraController? oldController = controller;
    if (oldController != null) {
      // `controller` needs to be set to null before getting disposed,
      // to avoid a race condition when we use the controller that is being
      // disposed. This happens when camera permission dialog shows up,
      // which triggers `didChangeAppLifecycleState`, which disposes and
      // re-creates the controller.
      controller = null;
      await oldController.dispose();
    }

    final CameraController cameraController = CameraController(
      cameraDescription,
      kIsWeb ? ResolutionPreset.max : ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    controller = cameraController;

    // If the controller is updated then update the UI.
    cameraController.addListener(() {
      if (mounted) {
        setState(() {});
      }
      if (cameraController.value.hasError) {
        showInSnackBar(
            'Camera error ${cameraController.value.errorDescription}');
      }
    });

    try {
      await cameraController.initialize();
      await Future.wait(<Future<Object?>>[
        // The exposure mode is currently not supported on the web.
        ...!kIsWeb
            ? <Future<Object?>>[
                cameraController.getMinExposureOffset().then(
                    (double value) => _minAvailableExposureOffset = value),
                cameraController
                    .getMaxExposureOffset()
                    .then((double value) => _maxAvailableExposureOffset = value)
              ]
            : <Future<Object?>>[],
        cameraController
            .getMaxZoomLevel()
            .then((double value) => _maxAvailableZoom = value),
        cameraController
            .getMinZoomLevel()
            .then((double value) => _minAvailableZoom = value),
      ]);
    } on CameraException catch (e) {
      switch (e.code) {
        case 'CameraAccessDenied':
          showInSnackBar('You have denied camera access.');
          break;
        case 'CameraAccessDeniedWithoutPrompt':
          // iOS only
          showInSnackBar('Please go to Settings app to enable camera access.');
          break;
        case 'CameraAccessRestricted':
          // iOS only
          showInSnackBar('Camera access is restricted.');
          break;
        default:
          _showCameraException(e);
          break;
      }
    }

    if (mounted) {
      setState(() {});
    }
  }

  void onTakePictureButtonPressed() {
    takePicture().then((XFile? file) {
      if (mounted) {
        setState(() {
          imageFile = file;
        });
        if (file != null) {
          //showInSnackBar('Picture saved to ${file.path}');
          Navigator.of(context).push(MaterialPageRoute(builder: (context) {
            return SubmitPage(
              image: imageFile!,
              serverAddr: serverAddr,
            );
          }));
        }
      }
    });
  }

  void onExposureModeButtonPressed() {
    if (_exposureModeControlRowAnimationController.value == 1) {
      _exposureModeControlRowAnimationController.reverse();
    } else {
      _exposureModeControlRowAnimationController.forward();
      _focusModeControlRowAnimationController.reverse();
    }
  }

  void onFocusModeButtonPressed() {
    if (_focusModeControlRowAnimationController.value == 1) {
      _focusModeControlRowAnimationController.reverse();
    } else {
      _focusModeControlRowAnimationController.forward();
      _exposureModeControlRowAnimationController.reverse();
    }
  }

  void onSetFlashModeButtonPressed(FlashMode mode) {
    setFlashMode(mode).then((_) {
      if (mounted) {
        setState(() {});
      }
      showInSnackBar('Flash mode set to ${mode.toString().split('.').last}');
    });
  }

  void onSetExposureModeButtonPressed(ExposureMode mode) {
    setExposureMode(mode).then((_) {
      if (mounted) {
        setState(() {});
      }
      showInSnackBar('Exposure mode set to ${mode.toString().split('.').last}');
    });
  }

  void onSetFocusModeButtonPressed(FocusMode mode) {
    setFocusMode(mode).then((_) {
      if (mounted) {
        setState(() {});
      }
      showInSnackBar('Focus mode set to ${mode.toString().split('.').last}');
    });
  }

  Future<void> setFlashMode(FlashMode mode) async {
    if (controller == null) {
      return;
    }

    try {
      await controller!.setFlashMode(mode);
    } on CameraException catch (e) {
      _showCameraException(e);
      rethrow;
    }
  }

  Future<void> setExposureMode(ExposureMode mode) async {
    if (controller == null) {
      return;
    }

    try {
      await controller!.setExposureMode(mode);
    } on CameraException catch (e) {
      _showCameraException(e);
      rethrow;
    }
  }

  Future<void> setExposureOffset(double offset) async {
    if (controller == null) {
      return;
    }

    setState(() {
      _currentExposureOffset = offset;
    });
    try {
      offset = await controller!.setExposureOffset(offset);
    } on CameraException catch (e) {
      _showCameraException(e);
      rethrow;
    }
  }

  Future<void> setFocusMode(FocusMode mode) async {
    if (controller == null) {
      return;
    }

    try {
      await controller!.setFocusMode(mode);
    } on CameraException catch (e) {
      _showCameraException(e);
      rethrow;
    }
  }

  Future<XFile?> takePicture() async {
    final CameraController? cameraController = controller;
    if (cameraController == null || !cameraController.value.isInitialized) {
      showInSnackBar('Error: select a camera first.');
      return null;
    }

    if (cameraController.value.isTakingPicture) {
      // A capture is already pending, do nothing.
      return null;
    }

    try {
      final XFile file = await cameraController.takePicture();
      return file;
    } on CameraException catch (e) {
      _showCameraException(e);
      return null;
    }
  }

  void _showCameraException(CameraException e) {
    _logError(e.code, e.description);
    showInSnackBar('Error: ${e.code}\n${e.description}');
  }
}

class _MediaSizeClipper extends CustomClipper<Rect> {
  final Size mediaSize;
  const _MediaSizeClipper(this.mediaSize);
  @override
  Rect getClip(Size size) {
    return Rect.fromLTWH(0, 0, mediaSize.width, mediaSize.height);
  }

  @override
  bool shouldReclip(CustomClipper<Rect> oldClipper) {
    return true;
  }
}

/// CameraApp is the Main Application.
class CameraApp extends StatelessWidget {
  /// Default Constructor
  const CameraApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MelaScan',
      theme: ThemeData(
        primaryColor: Color.fromARGB(255, 104, 75, 255),
      ),
      home: HomePage(),
    );
  }
}

List<CameraDescription> _cameras = <CameraDescription>[];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent, // transparent status bar
  ));

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Fetch the available cameras before initializing the app.
  try {
    WidgetsFlutterBinding.ensureInitialized();
    _cameras = await availableCameras();
  } on CameraException catch (e) {
    _logError(e.code, e.description);
  }
  runApp(const CameraApp());
}

/// This allows a value of type T or T? to be treated as a value of type T?.
///
/// We use this so that APIs that have become non-nullable can still be used
/// with `!` and `?` on the stable branch.
// TODO(ianh): Remove this once we roll stable in late 2021.
T? _ambiguate<T>(T? value) => value;
