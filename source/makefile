TARGET = DEM-MPH

#CC = mpicxx     
#CC = g++        
CC = nvc++	 
OBJE = DEM-MPH.o errorfunc.o log.o
CFLAGS  = -O3 -acc -Minfo=accel -gpu=cc80 -cuda
#CFLAGS	  =   -mp -acc -03 -Mibfo=accel -ta-Tesla,c80
#CFLAGS  = -O3 -fopenmp
#CFLAGS  = -O3
#CFLAGS  = -g -Wall
LDFLAGS =  -lm -lcublas -lcusparse
#LDFLAGS =  -lm


$(TARGET): $(OBJE)
	$(CC) $(CFLAGS) $(WFLAGS) -o $@ $(OBJE) $(LDFLAGS) 

.cpp.o:
	$(CC) $(CFLAGS) $(WFLAGS) -c $< 

clean:
	/bin/rm -f *.o *~ $(TARGET)  $(TARGET).exe 


main.o			:log.h
errorfunc.o		:errorfunc.h
log.o			:errorfunc.h log.h
