# --- Example Usage (Main Block) ---
if __name__ == "__main__":
    # Create a temporary directory for dummy files
    temp_dir = "temp_atf_data"
    output_csvs_dir = os.path.join(temp_dir, "labeled_csvs_from_batch") # Not directly used in this script but kept for context
    output_atf_dir = os.path.join(temp_dir, "downsampled_atf_output")

    # Clean up previous runs if they exist, to ensure a clean test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_csvs_dir, exist_ok=True) # Ensure this exists if other scripts expect it
    os.makedirs(output_atf_dir, exist_ok=True) # New output dir for downsampled ATFs

    # --- Dummy Data Content (Now with more precise time steps and current values) ---
    # To precisely match 600 Hz (dt = 1/600), we need more digits for time.
    # For current and voltage, reflect the observed precision from the snippet.
    
    # Generate precise time values for 600 Hz
    # Let's generate 16 points, 0 to 15*dt
    times_600hz_precise = [f"{(i / 600):.7f}" for i in range(16)] 
    # Use example current/voltage values, extending them slightly
    current_vals_600hz = [
        "-1.27563", "-1.23291", "-1.30920", "-1.15662", "-1.32446", "-1.05591", 
        "-1.15356", "-1.25732", "-1.30000", "-1.35000", "-1.40000", "-1.45000",
        "-1.50000", "-1.55000", "-1.60000", "-1.65000"
    ]
    voltage_vals_600hz = [
        "-69.6106", "-69.5801", "-69.6106", "-69.5496", "-69.6716", "-69.7327",
        "-69.7327", "-69.8242", "-69.8500", "-69.8700", "-69.8900", "-69.9000",
        "-69.9100", "-69.9200", "-69.9300", "-69.9400"
    ]
    dummy_data_rows_600hz = "\n".join([
        f"{times_600hz_precise[i]}\t{current_vals_600hz[i]}\t{voltage_vals_600hz[i]}" 
        for i in range(len(times_600hz_precise))
    ])

    dummy_raw_data_content_600hz = f"""ATF	1.0
7	3     
"AcquisitionMode=Gap Free"
"Comment="
"YTop=100.002,1000.02"
"YBottom=-100.002,-1000.02"
"SweepStartTimesMS=0.000"
"SignalsExported=Im_Scaled,10mV"
"Signals="	"Im_Scaled"	"10mV"
"Time (s)"	"Trace #1 (pA)"	"Trace #1 (mV)"
{dummy_data_rows_600hz}
"""

    # Generate precise time values for 500 Hz
    times_500hz_precise = [f"{(i / 500):.4f}" for i in range(13)] # dt = 0.0020
    current_vals_500hz = [
        "-1.30000", "-1.35000", "-1.40000", "-1.45000", "-1.50000", "-1.55000", 
        "-1.60000", "-1.65000", "-1.70000", "-1.75000", "-1.80000", "-1.85000", "-1.90000"
    ]
    voltage_vals_500hz = [
        "-69.7000", "-69.7100", "-69.7200", "-69.7300", "-69.7400", "-69.7500",
        "-69.7600", "-69.7700", "-69.7800", "-69.7900", "-69.8000", "-69.8100", "-69.8200"
    ]
    dummy_data_rows_500hz = "\n".join([
        f"{times_500hz_precise[i]}\t{current_vals_500hz[i]}\t{voltage_vals_500hz[i]}" 
        for i in range(len(times_500hz_precise))
    ])

    dummy_raw_data_content_500hz = f"""ATF	1.0
7	3     
"AcquisitionMode=Gap Free"
"Comment="
"YTop=100.002,1000.02"
"YBottom=-100.002,-1000.02"
"SweepStartTimesMS=0.000"
"SignalsExported=Im_Scaled,10mV"
"Signals="	"Im_Scaled"	"10mV"
"Time (s)"	"Trace #1 (pA)"	"Trace #1 (mV)"
{dummy_data_rows_500hz}
"""

    # Generate precise time values for 400 Hz
    times_400hz_precise = [f"{(i / 400):.4f}" for i in range(11)] # dt = 0.0025
    current_vals_400hz = [
        "-1.40000", "-1.45000", "-1.50000", "-1.55000", "-1.60000", "-1.65000", 
        "-1.70000", "-1.75000", "-1.80000", "-1.85000", "-1.90000"
    ]
    voltage_vals_400hz = [
        "-69.8000", "-69.8100", "-69.8200", "-69.8300", "-69.8400", "-69.8500",
        "-69.8600", "-69.8700", "-69.8800", "-69.8900", "-69.9000"
    ]
    dummy_data_rows_400hz = "\n".join([
        f"{times_400hz_precise[i]}\t{current_vals_400hz[i]}\t{voltage_vals_400hz[i]}" 
        for i in range(len(times_400hz_precise))
    ])

    dummy_raw_data_content_400hz = f"""ATF	1.0
7	3     
"AcquisitionMode=Gap Free"
"Comment="
"YTop=100.002,1000.02"
"YBottom=-100.002,-1000.02"
"SweepStartTimesMS=0.000"
"SignalsExported=Im_Scaled,10mV"
"Signals="	"Im_Scaled"	"10mV"
"Time (s)"	"Trace #1 (pA)"	"Trace #1 (mV)"
{dummy_data_rows_400hz}
"""

    # Create dummy files for different sampling rates
    input_files = []

    # File 1: 600 Hz (will be downsampled to 400 Hz)
    file_name_600 = "11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf"
    file_path_600 = os.path.join(temp_dir, file_name_600)
    with open(file_path_600, 'w') as f:
        f.write(dummy_raw_data_content_600hz)
    input_files.append(file_path_600)

    # File 2: 500 Hz (will be downsampled to 400 Hz)
    file_name_500 = "11d08003-guesthost_Trp-70_mV-500_Hz.atf"
    file_path_500 = os.path.join(temp_dir, file_name_500)
    with open(file_path_500, 'w') as f:
        f.write(dummy_raw_data_content_500hz)
    input_files.append(file_path_500)

    # File 3: 400 Hz (no downsampling needed)
    file_name_400 = "11e10002-guesthost_Phe-70_mV-400_Hz-rpt_2.atf"
    file_path_400 = os.path.join(temp_dir, file_name_400)
    with open(file_path_400, 'w') as f:
        f.write(dummy_raw_data_content_400hz)
    input_files.append(file_path_400)
    
    # Run the batch processor
    # Set view_plot to True to see the plots for downsampled files
    batch_process_data(input_files, target_sampling_rate=400, 
                       output_dir=output_atf_dir, view_plot=True,
                       log_file_name="my_downsampling_log.csv")

    print("\n--- Verifying final output ATF files ---")
    # Verify the created downsampled ATF files based on the *expected* naming convention
    # Note: Only 600Hz and 500Hz files are expected to be *created* in the output directory now
    expected_output_files_for_check = [
        os.path.join(output_atf_dir, "11n09001-guesthost_Tyr-70_mV-400_Hz-downsampled-rpt_1.atf"),
        os.path.join(output_atf_dir, "11d08003-guesthost_Trp-70_mV-400_Hz-downsampled.atf"),
    ]
    # The 400Hz file will NOT be in the output directory
    not_expected_in_output = os.path.join(output_atf_dir, "11e10002-guesthost_Phe-70_mV-400_Hz-downsampled-rpt_2.atf")


    for expected_file in expected_output_files_for_check:
        if os.path.exists(expected_file):
            print(f"Successfully created: {expected_file}")
            # Optional: Read back and check content
            try:
                times_out, current_out, voltage_out, header_out = load_atf(expected_file)
                print(f"  Loaded data points: {len(times_out)}")
                print(f"  Detected Hz from output file: {detect_sampling_rate(times_out)} Hz")
                print(f"  First 3 data points: Time={times_out[:3]}, Current={current_out[:3]}")
                # For closer inspection of precision
                if "600_Hz" in os.path.basename(expected_file) or "500_Hz" in os.path.basename(expected_file):
                    print(f"  First 3 time points (raw numbers): {times_out[:3]}")
                    print(f"  First 3 current points (raw numbers): {current_out[:3]}")
            except Exception as e:
                print(f"  Error verifying {expected_file}: {e}")
        else:
            print(f"FAILED to create: {expected_file} (This might be expected if it was not downsampled.)")

    if not os.path.exists(not_expected_in_output):
        print(f"Correctly NOT created (skipped): {not_expected_in_output}")
    else:
        print(f"ERROR: {not_expected_in_output} was created, but should have been skipped.")


    # Check the log file
    log_file_path_final = os.path.join(output_atf_dir, "my_downsampling_log.csv")
    if os.path.exists(log_file_path_final):
        print(f"\nLog file '{log_file_path_final}' created:")
        log_df = pd.read_csv(log_file_path_final)
        print(log_df)
    else:
        print(f"\nLog file '{log_file_path_final}' not found.")

    # --- Clean up dummy files and directories ---
    print("\n--- Cleaning up dummy files and directories ---")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print("Cleanup complete.")
