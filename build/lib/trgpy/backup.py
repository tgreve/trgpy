#
#
def convert_file_to_csv(in_file, out_file=None):
    """
        Reads in a file with EMG data and converts it into the EMG_vx.x.csv
        format, such that it can be appended to the database at some stage.

        The default is to save the converted file as [in_file].csv, unless
        a specific out_file name is provided.

        TODO:

        MODIFICATION HISTORY:
            29.07.2019: started
    """

    # Read in file
    try:
        df = pd.read_csv(in_file, delim_whitespace=True, dtype=str)
        columns = df.columns

        # Check that ID, LINE, IDV, and REF are specified
        proceed = False
        if "ID" in columns and "LINE" in columns and "IDV" in columns and "REF" in columns:
            proceed = True
        else:
            print("Error: File must contain ID, LINE, IDV, and REF keys.")
    except IOError:
        print("Error: File does not appear to exist.")


    if proceed:
        # Remove duplicates in input file and calculate length
        N = len(df)

        # Define df_out
        N_out = 1000
        df_out = pd.DataFrame({"ID": ["" for i in np.arange(0,N_out)],
            "ID_ALT": ["" for i in np.arange(0,N_out)],
            "IO": [1 for i in np.arange(0,N_out)],
            "YEAR": [-999 for i in np.arange(0,N_out)],
            "TYPE": ["" for i in np.arange(0,N_out)],
            "LINE": ["" for i in np.arange(0,N_out)],
            "Z_OPT": [-999 for i in np.arange(0,N_out)],
            "ERR_Z_OPT": [-999 for i in np.arange(0,N_out)],
            "Z_LINE": [-999 for i in np.arange(0,N_out)],
            "ERR_Z_LINE": [-999 for i in np.arange(0,N_out)],
            "FWHM": [-999 for i in np.arange(0,N_out)],
            "ERR_FWHM": [-999  for i in np.arange(0,N_out)],
            "IDV": [-999 for i in np.arange(0,N_out)],
            "ERR_IDV": [-999 for i in np.arange(0,N_out)],
            "MAG": [-999 for i in np.arange(0,N_out)],
            "ERR_MAG": [-999 for i in np.arange(0,N_out)],
            "REF": ["" for i in np.arange(0,N_out)],
            "COMMENTS": ["www.digame-db.online" for i in np.arange(0,N_out)],
            "LIR_LIT": [-999 for i in np.arange(0,N_out)],
            "ERR_LIR_LIT": [-999 for i in np.arange(0,N_out)],
            "MSTAR": [-999 for i in np.arange(0,N_out)],
            "GOOD_FIT": ["" for i in np.arange(0,N_out)],
            "LIR_CIGALE": [-999 for i in np.arange(0,N_out)],
            "LFIR_CIGALE": [-999 for i in np.arange(0,N_out)],
            "REF_URL": ["" for i in np.arange(0,N_out)],
            "NED_URL": ["" for i in np.arange(0,N_out)]})

        # Fill out df_out with df
        columns = ["ID","ID_ALT","IO","YEAR","TYPE","LINE","Z_OPT","ERR_Z_OPT",
            "Z_LINE","ERR_Z_LINE", "FWHM","ERR_FWHM","IDV","ERR_IDV","MAG",
            "ERR_MAG","REF","COMMENTS","LIR_LIT","ERR_LIR_LIT", "MSTAR",
            "GOOD_FIT","LIR_CIGALE","LFIR_CIGALE","REF_URL","NED_URL"]

        for clmn in columns:
            if clmn in df.columns:
                df_out[clmn] = df[clmn]


        # Fix references so that they comply with Smith et al. (XXXX)
        df_out["REF"] = df_out["REF"].str.replace("+"," et al. (")
        df_out["REF"] = df_out["REF"].astype(str)+")"


        # Trim df_out, remove duplicate entries, and save as .csv file.
        df_out = df_out[0:N-1]
        df_out = df_out.drop_duplicates(["ID","LINE","REF"])
        if out_file:
            df_out.to_csv(out_file, index=False)
        else:
            df_out.to_csv(in_file[:-4]+".csv", index=False)





#
#
def add_csv_to_emg(in_file, out_file=None):
    """
        Reads in a csv file (in_file) with EMG data and adds it to the end of
        EMG_vx.x.csv database file. Unless an out_file is specified in which case,
        final csv is saved to that file name (out_file).

        -Reads in .csv file to be appended and master .csv file
        -Checks for duplicate rows for ID, LINE, IDV, REF
        -Adds empty row for every 2nd row in input .csv
        -Finds overlap between input and master .csv files and drops
         overlap entries from input .csv.
        -Appends input .csv to master .csv
        -Saves master .csv to file

        MODIFICATION HISTORY:
            31.07.2019: started
            06.08.2019: basically works
    """

    # Read in csv file to be appended
    try:
        df = pd.read_csv(in_file)

        # Drop duplicate rows (ID .and. LINE .and. REF)
        df = df.drop_duplicates(["ID","LINE","IDV","REF"])

        # Add empty row after each entry
        s = pd.Series("", df.columns)
        f = lambda d: d.append(s, ignore_index=True)
        grp = np.arange(len(df))
        df =  df.groupby(grp, group_keys=False).apply(f).reset_index(drop=True)

    except IOError:
        print("Error: File does not appear to exist.")


    # Read in master .csv file
    try:
        df_master = pd.read_csv("/Users/tgreve/Dropbox/Work/EMGs/web/test-scripts/EMGs-test.csv", names=["ID",
            "ID_ALT","IO","YEAR","TYPE","LINE","Z_OPT","ERR_Z_OPT",
            "Z_LINE","ERR_Z_LINE", "FWHM","ERR_FWHM","IDV","ERR_IDV","MAG",
            "ERR_MAG","REF","COMMENTS","LIR_LIT","ERR_LIR_LIT", "MSTAR",
            "GOOD_FIT","LIR_CIGALE","LFIR_CIGALE","REF_URL","NED_URL"])

        # Append an empty row at the end of dataframe
        #df_master.append(pd.Series("", df_master.columns), ignore_index=True)

    except IOError:
        print("Error: File does not appear to exist.")



    # Find overlapping entries and drop them from df
    df_merge = pd.merge(df_master, df, on=["ID", "LINE", "REF"], how='inner')
    df = df[~df["ID"].isin(df_merge["ID"])]

    # Append df to df_master --> df_final_master
    df_final_master = df_master.append(df, ignore_index = True)

    # Save file
    if out_file:
        df_final_master.to_csv(out_file, index=False)
    else:
        #df_final_master.to_csv(in_file[:-4]+".csv", index=False)
        df_final_master.to_csv("gnyf.csv", index=False)



