
class UFloat ( ):

    def __init__(self, init_value, init_unc ):
        self.n = init_value
        self.s = init_unc

    # def __mul__(self, multfor ):
    #     output = TransMatrix( 0 )
    #     if( isinstance( multfor, TransMatrix ) ):
    #         output.M = self.M * multfor.M
    #     else:
    #         output.M = self.M * multfor
    #     return output

    # def __truediv__(self, divfor ):
    #     output = TransMatrix( 0 )
    #     output.M = self.M / divfor
    #     return output
    
    # def __add__(self, multfor ):
    #     output = self.copy()
    #     if( isinstance( multfor, TransMatrix ) ):
    #         output.M = output.M + multfor.M
    #     else:
    #         output.M = output.M + multfor
    #     return output