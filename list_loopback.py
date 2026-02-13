# list_loopback.py
import pyaudiowpatch as pyaudio

def main():
    p = pyaudio.PyAudio()

    print("=== Host APIs ===")
    for i in range(p.get_host_api_count()):
        api = p.get_host_api_info_by_index(i)
        print(f"[{i}] {api['name']}")

    print("\n=== Devices (look for 'loopback' / duplicated WASAPI devices) ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "")
        host = p.get_host_api_info_by_index(info["hostApi"]).get("name", "")
        max_in = info.get("maxInputChannels", 0)
        max_out = info.get("maxOutputChannels", 0)
        sr = info.get("defaultSampleRate", 0)
        tag = ""
        low = name.lower()
        if "loopback" in low:
            tag = " <== LOOPBACK?"
        print(f"[{i:3d}] in={max_in} out={max_out} sr={sr:.0f} host={host}  name={name}{tag}")

    p.terminate()

if __name__ == "__main__":
    main()
