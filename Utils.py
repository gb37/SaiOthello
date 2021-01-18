



def LeggiXOT(FilePath, NLinea):
    
    Lette = 0
    with open(FilePath) as file:
        while True:
            read_data = file.readline()
                
            if len(read_data) < 16:
                file.close()
                return None
            
            Lette += 1

            if Lette >= NLinea:
                return read_data.rstrip()

             