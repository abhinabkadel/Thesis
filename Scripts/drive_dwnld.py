# %%
import os
import pandas as pd

# %%
src_rt      = r"wfrt:/reforecasts-2/"
src_dir     = pd.date_range(start='20140701', end='20140715').strftime("%Y%m%d.%H").values
dst_rt      = r"D:\Masters\Thesis\Test_downloads"
# rclone_cmd  = "rclone copy --verbose --update --progress --dry-run "
rclone_cmd  = "rclone copy --verbose --update --progress "
# other options --interactive 


# %%
for folder in src_dir:
    print(folder)
    print("\n")
    for i in [*range(1, 16), 52]:
        print(i)
        fname = f"Qout_south_asia_mainland_{i:d}.nc"
        src_pth = src_rt + folder + "/" + fname
        dst_pth = os.path.join(dst_rt, folder[:-3])
        os.system(rclone_cmd + src_pth + " " + dst_pth)
    

# %%
