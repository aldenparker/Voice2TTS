"""Spike 1: enumerate audio devices and find candidate virtual-cable outputs."""

import sounddevice as sd

CABLE_HINTS = ("cable", "voicemeeter", "virtual", "vb-audio")


def main() -> None:
    hostapis = sd.query_hostapis()
    print(f"{'idx':>4}  {'in':>3} {'out':>3}  {'rate':>6}  hostapi          name")
    print("-" * 92)
    for idx, dev in enumerate(sd.query_devices()):
        api = hostapis[dev["hostapi"]]["name"]
        print(
            f"{idx:>4}  {dev['max_input_channels']:>3} {dev['max_output_channels']:>3}"
            f"  {int(dev['default_samplerate']):>6}  {api:<15}  {dev['name']}"
        )

    print()
    din, dout = sd.default.device
    print(f"default input : {din} {sd.query_devices(din)['name'] if din is not None else ''}")
    print(f"default output: {dout} {sd.query_devices(dout)['name'] if dout is not None else ''}")

    print("\nvirtual-cable candidates (outputs):")
    found = False
    for idx, dev in enumerate(sd.query_devices()):
        name = dev["name"].lower()
        if dev["max_output_channels"] > 0 and any(h in name for h in CABLE_HINTS):
            api = hostapis[dev["hostapi"]]["name"]
            print(f"  [{idx}] {dev['name']}  ({api})")
            found = True
    if not found:
        print("  none -- VB-CABLE is not installed yet.")


if __name__ == "__main__":
    main()
