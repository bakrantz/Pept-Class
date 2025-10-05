# --- Example Usage (Main Block) ---
if __name__ == "__main__":
    input_filepaths_list= [
        "./guesthost_Tyr/11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf",
        "./guesthost_Tyr/11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf",
        "./guesthost_Tyr/11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf",
        "./guesthost_Tyr/11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf",
        "./guesthost_Tyr/11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf",
        "./guesthost_Tyr/11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf",
        "./guesthost_Tyr/11n09001-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf",
        "./guesthost_Tyr/11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf",
        "./guesthost_Tyr/11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf",
        "./guesthost_Tyr/11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf",
        "./guesthost_Tyr/11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_4.atf",
        "./guesthost_Tyr/11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_5.atf",
        "./guesthost_Tyr/11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_6.atf",
        "./guesthost_Tyr/11n09002-guesthost_Tyr-70_mV-600_Hz-rpt_7.atf",
        "./guesthost_Tyr/11n09003-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf",
        "./guesthost_Tyr/11n09003-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf",
        "./guesthost_Tyr/11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf",
        "./guesthost_Tyr/11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf",
        "./guesthost_Tyr/11n09004-guesthost_Tyr-70_mV-600_Hz-rpt_3.atf",
        "./guesthost_Tyr/11n16003-guesthost_Tyr-70_mV-600_Hz-rpt_1.atf",
        "./guesthost_Tyr/11n16003-guesthost_Tyr-70_mV-600_Hz-rpt_2.atf"
        ]
    target_sampling_rate = 400
    batch_process_data(input_filepaths_list,
                       target_sampling_rate = target_sampling_rate,
                       output_dir = "./guesthost_Tyr/",
                       raw_header_idx = 9,
                       view_plot = False,
                       log_file_name = "downsampling_log.csv")
