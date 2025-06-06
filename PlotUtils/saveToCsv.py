from os.path import exists
import pandas as pd
from glob import glob

def getFilename( id, basename ):
    return f"{basename}{id:03d}"

def getNextFilename( basename ):
    counter = 1
    while( len( glob( getFilename( counter, basename ) + "*" ) ) > 0 ):
        counter += 1
    return getFilename( counter, basename )

def saveToCsv( data, basename = "results/r" ):
    df = pd.DataFrame.from_dict( data )

    filename = getNextFilename( basename )

    df.to_csv( filename + ".csv", index = False )

    print( f"Saved as {filename}")

    return filename