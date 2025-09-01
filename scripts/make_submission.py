import shutil, os

src = os.path.join("results", "bootcamp_submission.csv")
dst = os.path.join("results", "submission.csv")

shutil.copy(src, dst)
print(f" Arquivo gerado: {dst}")
