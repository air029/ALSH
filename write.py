def write(file, text):
    f = open(file, mode = 'a')
    f.writelines(text)
    f.close()

