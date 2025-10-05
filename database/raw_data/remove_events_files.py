import os, re

# Script to remove Clampfit annotated events which are not needed
# since K-Means was used to label state instead in this dataset

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            print(f"Would delete: {os.path.join(dir, f)}")
#           os.remove(os.path.join(dir, f))

if __name__ == "__main__":

    dirs = [
        './PA/guesthost_Leu/',
        './PA/guesthost_Thr/',
        './PA/guesthost_TrpDL/',
        './PA/guesthost_Ala/',
        './PA/guesthost_Phe/',
        './PA/guesthost_Trp/',
        './PA/guesthost_Tyr/'
    ]

    for dir in dirs:
        purge(dir, '_events.atf')


        
