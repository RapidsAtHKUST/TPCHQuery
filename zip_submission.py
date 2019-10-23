import datetime
import os

if __name__ == '__main__':
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    print(date_str)
    os.system('zip -r tpch-rapids-{}.zip . -x *.DS_Store -x /raw* -x *cmake-build-debug* -x *.idea* -x *.py -x *.git* -x tpch-q3-submit'.format(date_str))
