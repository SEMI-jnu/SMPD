import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run ScanCode Toolkit to extract SBOM.")
    parser.add_argument("--input", help="Directory containing package folders to scan")
    parser.add_argument("--output", help="Directory to save SBOM JSON files")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(os.path.join(base_dir, "src/config"))
    from pipeline_config import SCANCODE_EXECUTABLE, SCANCODE_TIMEOUT, PATHS

    scancode_path = SCANCODE_EXECUTABLE
    input_dir = args.input if args.input else os.path.join(base_dir, PATHS["raw_packages"])
    output_dir = args.output if args.output else os.path.join(base_dir, PATHS["sbom_output"])
    timeout_limit = SCANCODE_TIMEOUT

    os.makedirs(output_dir, exist_ok=True)

    scan_options = ["--package", "--license", "--copyright", "--info", "--classify", "--summary", "--json-pp"]

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    if not os.path.exists(input_dir):
        print("Error: Input package directory does not exist.")
        return

    packages = [pkg for pkg in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, pkg))]
    print(f"Found {len(packages)} packages to process.")

    for package_name in packages:
        package_path = os.path.join(input_dir, package_name)
        output_file = os.path.join(output_dir, f"{package_name}.json")

        if os.path.exists(output_file):
            print(f"Skip: Already scanned ({package_name})")
            continue

        cmd = [scancode_path] + scan_options + [output_file, package_path]
        print(f"Scanning: {package_name} ...")

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            process.wait(timeout=timeout_limit)

            if process.returncode == 0:
                print(f" -> Success: {output_file}")
            else:
                print(f" -> Error: Scan failed ({package_name})\n{process.stderr.read()}")
        
        except subprocess.TimeoutExpired:
            print(f" -> Timeout: Process killed after {timeout_limit} seconds ({package_name})")
            process.kill()
        except Exception as e:
            print(f" -> Exception: An error occurred ({package_name}): {e}")

    print("\nAll packages have been scanned successfully.")

if __name__ == "__main__":
    main()
