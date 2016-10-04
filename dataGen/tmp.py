
fp = open("a.txt","w")
for elem in ["asdf",1,3,4]:
    print>>fp, elem
fp.close()
