# SO-101 independent cameras

OBS Studio's built-in Virtual Camera exports one selected program, scene or source. This implementation requests each OBS input source independently through obs-websocket v5 GetSourceScreenshot. It does not split a side-by-side composite. Omitting --camera-config keeps the old iphone virtual-camera path.

The local OBS scene contains separate iPad_Overhead and iPhone6_Fixed DroidCam OBS inputs. Their source UUIDs and Wi-Fi IP settings are in ignored cameras.local.yaml. The optional wrist input opens InnoMaker directly through AVFoundation by uniqueID `0x11240000c456366`, without relying on an OpenCV index. IP or USB reassignment requires reviewing the local YAML.

OBS 32 has obs-websocket built in. Enable it in OBS Tools → WebSocket Server Settings and keep password authentication enabled. Copy the project-root `.env.example` to `.env`, then enter the OBS WebSocket password after `OBS_WEBSOCKET_PASSWORD=`. The `.env` file is ignored by Git. A nonempty process environment variable takes precedence; without `.env`, the environment variable still works. Never put the password in camera YAML.

    cd /path/to/so101-imitation-learning
    conda activate lerobot
    export PYTHONPATH="$PWD/.local/camera-deps:$PWD/.local/act-deps:$PYTHONPATH"

The isolated websocket-client dependency is installed in .local/camera-deps. If needed elsewhere, use python -m pip install --no-deps --target .local/camera-deps -r scripts/so101_capture/requirements-multicam.txt.

Preview without a robot connection, using a different new output path for each invocation:

    python scripts/so101_capture/preview_cameras.py --camera-config scripts/so101_capture/cameras.local.yaml --enable-cameras overhead --output /tmp/so101-preview-1
    python scripts/so101_capture/preview_cameras.py --camera-config scripts/so101_capture/cameras.local.yaml --enable-cameras overhead,side --output /tmp/so101-preview-2
    python scripts/so101_capture/preview_cameras.py --camera-config scripts/so101_capture/cameras.local.yaml --enable-cameras overhead,side,wrist --output /tmp/so101-preview-3

Each preview writes one PNG per logical camera and preview.json with source identities and sampled-frame counts. Inspect actual images before recording. The local wrist camera is disabled by default; `--enable-cameras overhead,side,wrist` explicitly selects it. The native helper is compiled into `/tmp` on first use and requests the exact AVFoundation uniqueID. It scales USB frames to 320x240 before transfer; Python converts BGRA to RGB. AVFoundation callback-to-host latency above 100 ms is rejected, while the usual host-receipt freshness and 75 ms cross-camera skew checks remain active. For a two-camera fallback use `--enable-cameras overhead,side`. `--diagnose-stale` continues a robot-free benchmark while rejecting stale/skew samples; it does not relax the recording or inference checks.

With OBS's InnoMaker_Wrist source still active, three 40-second direct-wrist preview attempts did not complete: the wrist callback-to-host latency exceeded 100 ms during startup (117-238 ms). This does not prove OBS owns the device exclusively; contention is one plausible cause. To isolate it, stop recording/streaming in OBS, deactivate the InnoMaker_Wrist source only (right-click source → Deactivate, if offered; otherwise temporarily remove it from the active scene without deleting it), leave the two DroidCam sources active, then rerun three normal previews. Restore the source afterward. Do not treat three-camera capture as qualified until those runs complete without faults. No OBS settings were changed by this implementation.

record_episode.py accepts --camera-config and optional --enable-cameras. It creates video features only for selected logical names, such as observation.images.overhead and observation.images.side. capture_timing.json records host-side camera response timestamps and Follower observation timestamps for each frame. OBS screenshot API does not expose the source's exposure time. The 100 ms freshness check, 75 ms between-camera sample skew check, and 10 s identical-image watchdog stop on suspected frozen sources. A truly static scene may also trigger the watchdog.

The existing resize script now resizes every enabled camera independently. A training command generator validates the Dataset feature keys:

    python scripts/so101_capture/resize_act_dataset.py SOURCE_DATASET NEW_168x224_DATASET --expected EPISODES
    python scripts/so101_capture/prepare_multicam_train.py --camera-config scripts/so101_capture/cameras.local.yaml --enable-cameras overhead,side,wrist --dataset NEW_168x224_DATASET --output NEW_TRAIN_OUTPUT

The second command prints an ACT training command and never trains. act_handoff_launcher.py accepts the same camera options. It checks checkpoint image keys and shapes before hardware initialization. A checkpoint trained with a different camera set is rejected.

OBS rendering and JPEG/WebSocket response time may prevent 20 Hz with three cameras. Throughput, visual identity, orientation, Wi-Fi freeze behavior and image synchronization need robot-free preview on this Mac before recording or ACT use.

On the current Mac, three OBS screenshots occasionally exceed the unchanged 100 ms freshness limit even when the 20-second average is 20 Hz. JPEG quality 70 reduced typical request/decode time but did not eliminate stale frames in two normal 20-second trials. The optional per-camera `jpeg_quality` YAML value defaults to 90 and does not change image dimensions; quality 70 is an experiment, not a qualified recording setting. The current next recording candidate is `overhead,side`, which completed repeated 20-second previews without a freshness stop. Recheck under the intended recording load before adopting a Dataset. Direct AVFoundation wrist capture would need a separate Unique-ID-bound capture path and a separate benchmark; OpenCV index selection is not a stable replacement.

The shared OBS screenshot resolution defaults to 640x480. To trial a smaller resolution without changing OBS or DroidCam, add a top-level `capture` mapping to a separate copy of the local YAML, for example `capture: {width: 320, height: 240}`. Supported pairs are 640x480, 448x336, and 320x240. The selected RGB shape is used directly for the capture Dataset and MP4; `resize_act_dataset.py` then makes a new 224x168 ACT Dataset, and `act_handoff_launcher.py` accepts the same live shape before resizing to 224x168. Never mix camera configurations or checkpoints trained with different image keys. Preview only, with a new output each time:

    python scripts/so101_capture/preview_cameras.py --camera-config /tmp/so101-cameras-320x240.yaml --enable-cameras overhead,side --seconds 20 --hz 20 --output /tmp/so101-preview-320-trial1

Two 20-second overhead+side previews passed for each tested size. A longer 448x336 preview stopped at 26.8 s on a 116 ms stale overhead image; one 320x240 40-second preview passed. This is evidence of reduced average cost, not proof of stable robot recording. Keep the 100 ms stale-frame rejection and validate the intended capture workload before using any new Dataset for training.

For a fixed-rate performance check, run the preview with `--seconds 20 --hz 20` and a new output directory. `preview.json` reports loop intervals, completed screenshots, reused responses, adjacent identical sampled images, per-camera image age, host-response skew, and request/base64/decode times. The camera sources use separate WebSocket connections and background threads; the preview loop samples their latest frames at 20 Hz. Request time combines OBS rendering/JPEG encoding and WebSocket transfer, which this API cannot separate. Identical images may mean an unchanged scene, and a fresh screenshot response does not prove a fresh sensor exposure. Compare `capture_timing.json` image response timestamps with `observed_at` after a future recording to assess host-side State/image offset; exposure-time synchronization cannot be proven from these timestamps.

Sources: OBS Virtual Camera https://obsproject.com/kb/virtual-camera-guide ; OBS WebSocket https://github.com/obsproject/obs-websocket/blob/master/README.md ; DroidCam OBS multi-device plugin https://github.com/dev47apps/droidcam-obs-plugin/blob/master/README.md .

## Device identity before a dataset run

At connection and about every two seconds, each logical name is checked against its OBS source name, source UUID, and configured underlying-device setting. For Wi-Fi DroidCam sources this is `cur_dev_ip`. A changed source UUID or configured IP fails closed; after a legitimate DHCP change, update the ignored local YAML only after visually confirming the device. An OBS restart is acceptable only if it reloads the same scene/source UUIDs and devices; rerun the robot-free preview after restart and inspect each PNG. This has not been established by restarting OBS automatically.

The OBS source UUID identifies the source, **not** the iPad or iPhone hardware. An unchanged configured IP does not prove which physical device supplies its pixels: DHCP address reuse, an upstream feed swap, a frozen image, or a wrong physical mounting angle can escape the metadata checks. Freeze detection is delayed and static scenes can look frozen. Inspect the actual overhead/side/wrist images before each recording and inference run, and keep OBS/device assignments fixed during an episode. An episode interrupted by a source fault must not be adopted for training.

For the USB wrist camera, OBS's macOS AVFoundation capture source stores `AVCaptureDevice.uniqueID` in its `device` setting. The current local setting was independently matched against AVFoundation's device list. Do not rely on model name or OpenCV index. This identifies a device on this Mac, but does not certify image content or mounting. Three-camera recording still requires a stable freshness benchmark.
