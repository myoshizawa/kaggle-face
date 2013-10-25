import numpy as np
import math



#This function computes the integral image. 
#INPUT: An NxN numpy array representing an image
#OUTPUT: The corresponding integral image
def integral_image(data):
 
 N=int(math.sqrt(data.size))
 s = np.array((np.zeros(data.size))).reshape(N,N)
 int_im=np.array((np.zeros(data.size))).reshape(N,N)
 s[0][0]=data[0][0]
 int_im[0][0]=data[0][0]
 
 for x in xrange(1,N):
  s[0][x] = s[0][x-1]+ data[0][x]
  int_im[0][x]=s[0][x]
  s[x][0] = data[x][0]	 
  int_im[x][0]=int_im[x-1][0] + s[x][0]
 
 
 for x in xrange(1,N):
   for y in xrange(1,N):
     s[x][y] = s[x][y-1] + data[x][y]
     int_im[x][y] = int_im[x-1][y] + s[x][y]
 	 
 return  int_im


 
#DESCRIPTION: Given an integral image this function is capable of computing
 #any kind of feature.
##INPUT: The type of feature, be it 12,21,13,31 or 22. The column and row of 
#the top left corner as x1,y1 and bottom right corner as x2,y2. Finally, it takes
#the integral image as well.
##OUTPUT: The desired feature
 
 
def rect_features(type,x1,y1,x2,y2,ii):
 #2-features parted vertically
 if type == 12:
  
  mid_y=(y2+y1-1)/2
   
   #determining key points in integral image:
  if (x1==0 or y1==0):
    s1=0
  else:
    s1=ii[x1-1][y1-1]
  if x1==0:
    s2=0
    s3=0
  else:
    s2=ii[x1-1][mid_y]
    s3=ii[x1-1][y2]
   
  if y1==0:
    s4=0
  else:
   s4=ii[x2][y1-1]
    
  s5=ii[x2][mid_y]
  s6=ii[x2][y2]      
     
     #computing final value
  return (-s1+2*s2-s3+s4-2*s5+s6)
  
  
  #2-features sliced horizontally
 elif type == 21:
  
  #determining key points 
  mid_x=(x1+x2-1)/2
  if (x1==0 or y1==0):
    s1=0
  else:
    s1=ii[x1-1][y1-1]
  if x1==0:
    s2=0
  else:
    s2=ii[x1-1][y2]
      
  if y1==0:
    s3=0
    s5=0
  else:
    s3=ii[mid_x][y1-1]
    s5=ii[x2][y1-1]
    
  s4=ii[mid_x][y2]
  s6=ii[x2][y2]    
  
  #computing final value
  return (s1-s2-2*s3+2*s4+s5-s6)
  
  #3-features sliced vertically
 elif type == 13:
  
  third_y=(y2-y1+1)/3
  
  #computing key points
  if(x1==0 or y1==0):
   s1=0
  else:
   s1=ii[x1-1][y1-1]
   
  if x1==0:
   s2=0
   s3=0
   s4=0
  else:
   s2=ii[x1-1][y1+third_y-1]
   s3=ii[x1-1][y1+2*third_y-1]
   s4=ii[x1-1][y2]
  if y1==0:
   s5=0
  else:
   s5=ii[x2][y1-1]
   
  s6=ii[x2][y1+third_y-1]
  s7=ii[x2][y1+2*third_y-1]
  s8=ii[x2][y2]      
  
  #final value
  return (-s1+2*s2-2*s3+s4+s5-2*s6+2*s7-s8)

  #3-features sliced horizontally
 elif type == 31:
  
  third_x=(x2-x1+1)/3
  
  #computing key points
  if(x1==0 or y1==0):
   s1=0
  else:
   s1=ii[x1-1][y1-1]
   
  if x1==0:
   s2=0
  else:
   s2=ii[x1-1][y2]
   
  if y1==0:
   s3=0
   s5=0
   s7=0
  else:
   s3=ii[x1+third_x-1][y1-1]
   s5=ii[x1+2*third_x-1][y1-1]
   s7=ii[x2][y1-1]
   
  s4=ii[x1+third_x-1][y2]
  s6=ii[x1+2*third_x-1][y2]
  s8=ii[x2][y2]       
  
  #final value
  return (-s1+s2+2*s3-2*s4-2*s5+2*s6+s7-s8)

 #4-features. 
 elif type == 22:
 
  mid_x=(x1+x2-1)/2
  mid_y=(y1+y2-1)/2
 
  #finding key points
  if(x1==0 or y1==0):
   s1=0
  else:
   s1=ii[x1-1][y1-1]
   
  if x1==0:
   s2=0
   s3=0
  else:
   s2=ii[x1-1][mid_y]
   s3=ii[x1-1][y2]
   
  if y1==0:
   s4=0
   s7=0
  else:
   s4=ii[mid_x][y1-1]
   s7=ii[x2][y1-1]
   
  s5=ii[mid_x][mid_y]
  s6=ii[mid_x][y2]
  s8=ii[x2][mid_y]
  s9=ii[x2][y2]  
    
    #final result
  return (-s1+2*s2-s3+2*s4-4*s5+2*s6-s7+2*s8-s9)
 
 