import os, json, sys
version = "7.13.01_4.181.1147"
#trail_dir = "/Users/pestun/cur/Trail"
#trail_dir = "/Users/pestun/temp"

print("usage "  + sys.argv[0] + " [trail_base_dir]")

if len(sys.argv) == 2:
    trail_dir =  sys.argv[1]
else:
    print("assuming trail_base_dir is the current dir ", os.getcwd())
    trail_dir = os.getcwd()


data_dir = "data"
mptp2_articles_dir = os.path.join(trail_dir, data_dir, "MPTP2", 
                              version, "problems_small_consist")

mizar_dir = os.path.join(trail_dir, data_dir, "mizar-" + version,
                      "mizshare", "mml")

text_info_name = os.path.join(trail_dir, data_dir,"MPTP2", version,
                              "mptp2_mizar_map.txt")
json_info_name = os.path.join(trail_dir, data_dir,"MPTP2", version,
                              "mptp2_mizar_map.json")

mptp2_articles = set(x for x in os.listdir(mptp2_articles_dir) 
                     if os.path.isdir(os.path.join(mptp2_articles_dir,x)))
mizar_articles = set(x[:-4] for x in os.listdir(mizar_dir))

print("Articles in mizar that are not in mptp2:",
      mizar_articles - mptp2_articles)

print("Articles in mptp2 that are not in mizar:",
      mptp2_articles - mizar_articles)





# for each mptp2 article get all problems there and extract mml info           
# from the first line

info_table = dict()
n_articles = len(mptp2_articles)
for cur, article in enumerate(list(mptp2_articles)):
    print("processing mizar article", article, cur+1, "out of ",
          n_articles, end='\r')
    for problem in sorted(os.listdir(os.path.join(mptp2_articles_dir, article))):
        article_name, problem_name=problem.split('__')
        assert article == article_name, "inconsistency between " \
                            "article_name and problem_name" 
        error_msg = "can't parse the mizar info line in " \
                                     + article + "/" + problem
        with open(os.path.join(mptp2_articles_dir, article, problem), 'r') as f:
            mizar_info_string = f.readline()
        
        pattern = problem_name + "," + article_name
        pattern_s = mizar_info_string.find(pattern)
        assert pattern_s != -1, error_msg
        
        pattern_f = pattern_s + len(pattern)
        location_pair = mizar_info_string[pattern_f+1:].split(',')
        assert len(location_pair) == 2, error_msg
        
        location_line  = location_pair[0]
        assert str.isdigit(location_line), error_msg
        
        mizar_line = int(location_line)
        
        #print(article_name, problem_name, mizar_line)
        with open(os.path.join(mizar_dir, article + '.miz'), 'r',
                  encoding='latin1') as f:
            article_lines = f.readlines()
            
        article_pretext = "".join(article_lines[:mizar_line-1])
        last_end_point = article_pretext.rfind(';')
        
        mizar_extract = article_pretext[last_end_point+1:] + article_lines[mizar_line-1]
        
        mizar_extract = mizar_extract.strip().split('\n')
        
        info_table[problem_name] = (article_name, mizar_line, mizar_extract)

print("")        
print('building MPTP2 to Mizar info table is completed')

print('dumping MPTP2 to Mizar info table to JSON', json_info_name)
with open(json_info_name,'w') as f:
    json.dump(info_table, f)





info_list = list(info_table.items())

info_list = sorted(info_list, key = lambda x: x[0])
info_list = sorted(info_list, key = lambda x: x[1])
print('dumping the MPTP2 to Mizar info table to TXT', text_info_name)
with open(text_info_name, 'w') as f:
    for i in info_list:
        f.write(i[0] + ', ' + i[1][0] + ', ' + str(i[1][1]) + '\n')
        for l in i[1][2]:
            f.write('# ' + l + '\n')
        f.write('\n')
        

        

        
    
        

