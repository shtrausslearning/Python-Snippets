import re

# function to read pvcc format & define camera position for pyvista Plotter()
def read_pvcc(name):

    f = open(name,"r")
    ii=-1

    # loop all rows
    for i in f:
        lstr = i; ii+=1
        lstr_filter = re.findall(r'"([^"]*)"', lstr)  # find string in quotation marks

        # camera position
        if(ii is 4):
            pos_x = float(lstr_filter[1])
        if(ii is 5):
            pos_y = float(lstr_filter[1])
        if(ii is 6):
            pos_z = float(lstr_filter[1]) 

        # camera focal point
        if(ii is 9):
            fp_x = float(lstr_filter[1]) 
        if(ii is 10):
            fp_y = float(lstr_filter[1]) 
        if(ii is 11):
            fp_z = float(lstr_filter[1])  

        # camera viewup 
        if(ii is 14):
            vu_x = float(lstr_filter[1]) 
        if(ii is 15):
            vu_y = float(lstr_filter[1]) 
        if(ii is 16):
            vu_z = float(lstr_filter[1])

    # return list of tuples
    return [(pos_x,pos_y,pos_z),
                (fp_x,fp_y,fp_z),
                 (vu_x,vu_y,vu_z)]

    f.close()
