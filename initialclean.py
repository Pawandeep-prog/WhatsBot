import re
def clean():
    """
    Cleans the path of words containing <omitted> and emoji's 
    """
    path = 'chat.txt'
    f = open(path,'r',errors="ignore")
    lines = f.readlines()

    pattern1 = r'.*<\w+ omitted>'
    pattern2 = r'[^\x00-\x7f]+'
    for i in lines:
        if re.search(pattern1, i):
            lines.remove(i)

    f.close()

    f = open(path,'w',errors="ignore")
    for i in lines:
        f.write(re.sub(pattern2, '', i))  #to ignore non-ascii characters
    f.close()
