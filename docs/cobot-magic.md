# Cobot Magic Remote Runtime

TiPToP runs on the GPU machine. Cobot Magic's `tiptop_client` runs on the robot upper computer, where it is the only component that uses ROS or accesses the RealSense camera. TiPToP uses its existing ZeroMQ REQ/REP + MessagePack/NumPy bridge; it does not start ROS or open a camera directly.

## Start the upper-computer bridge

Start ROS, the arm driver, and the RealSense drivers as documented by `Cobot_Magic/tiptop_client/README.md`. Then start its two bridge services on the upper computer:

```bash
python3 -m tiptop_client controller-server --config tiptop_client/config.yaml
python3 -m tiptop_client camera-server --config tiptop_client/config.yaml
```

The default bridge ports are controller `5555` and camera `5556`. Bind them only to a trusted interface. A TiPToP process should normally use SSH forwarding rather than expose either port publicly.

## Forward the two RPC services

From the GPU machine, use local forwarding (replace `agilex@upper-computer`):

```bash
ssh -N \
  -L 15555:127.0.0.1:5555 \
  -L 15556:127.0.0.1:5556 \
  agilex@upper-computer
```

Configure the matching local tunnel addresses, distinct ports, camera serial, and timeouts in `tiptop/config/tiptop.yml`:

```yaml
robot:
  type: cobot_magic
  dof: 6
  controller_host: "127.0.0.1"
  controller_port: 15555
  request_timeout_ms: 30000
  trajectory_timeout_ms: 300000
  max_message_bytes: 134217728

cameras:
  hand:
    type: remote_realsense
    serial: "339222070351"
    camera_host: "127.0.0.1"
    camera_port: 15556
    request_timeout_ms: 30000
    max_message_bytes: 134217728
```

## Verify the bridge

These commands use the configured tunnel values unless explicitly overridden:

```bash
cobot-controller-health
cobot-camera-health
```

The controller client sends one high-level six-joint trajectory at a time and never retries a command after a timeout or connection error. The remote camera request must name its configured `serial`; it returns RGB, the IR1/IR2 pair, and calibration values. It does not provide depth. TiPToP preserves its FoundationStereo depth-estimation path for remote Cobot Magic cameras.

## OmniGround

Run OmniGround where its configured model is available, for example:

```bash
cd ../OmniGround
pixi run server -- --host 127.0.0.1 --port 8011 --model-id molmo2-er
```

Set `perception.vlm.url`, `endpoint` (`/generate` or `/v1/generate`), required `model_id`, optional `temperature`, and `timeout_seconds` in `tiptop/config/tiptop.yml`. TiPToP sends a multipart PNG image, the complete prompt, and that model ID. OmniGround must return the direct `{"bboxes": [...], "predicates": [...]}` JSON object.
