import os
import shutil

if __name__ == '__main__':
    rootPath = "/home/hya/Downloads/20bn-jester-v1"
    with open("jester-v1-validation.csv", "r") as f:
        info = f.readlines()
        for inf in info:
            num = inf.split(";")[0]
            real_path = os.path.join(rootPath, num)
            dest_path = os.path.join("validation", num)
            shutil.move(real_path, dest_path)

    print("complete")