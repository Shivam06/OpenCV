class sudoku:

    def __init__(self, grid):
        self.arr = grid
    
    def print_grid(self):
        for i in range(9):
            for j in range(9):
                print self.arr[i][j],
            print ('\n')
     
             
    # Function to Find the entry in the Grid that is still  not used
    # Searches the grid to find an entry that is still unassigned. If
    # found, the reference parameters row, col will be set the location
    # that is unassigned, and true is returned. If no unassigned entries
    # remain, false is returned.
    # 'l' is a list  variablethat has been passed from the solve_sudoku function
    # to keep track of incrementation of Rows and Columns
    def find_empty_location(self,l):
        for row in range(9):
            for col in range(9):
                if(self.arr[row][col]==0):
                    l[0]=row
                    l[1]=col
                    return True
        return False
     
    # Returns a boolean which indicates whether any assigned entry
    # in the specified row matches the given number.
    def used_in_row(self,row,num):
        for i in range(9):
            if(self.arr[row][i] == num):
                return True
        return False
     
    # Returns a boolean which indicates whether any assigned entry
    # in the specified column matches the given number.
    def used_in_col(self,col,num):
        for i in range(9):
            if(self.arr[i][col] == num):
                return True
        return False
     
    # Returns a boolean which indicates whether any assigned entry
    # within the specified 3x3 box matches the given number
    def used_in_box(self,row,col,num):
        for i in range(3):
            for j in range(3):
                if(self.arr[i+row][j+col] == num):
                    return True
        return False
     
    # Checks whether it will be legal to assign num to the given row,col
    #  Returns a boolean which indicates whether it will be legal to assign
    #  num to the given row,col location.
    def check_location_is_safe(self,row,col,num):
         
        # Check if 'num' is not already placed in current row,
        # current column and current 3x3 box
        return not self.used_in_row(row,num) and not self.used_in_col(col,num) and not self.used_in_box(row - row%3,col - col%3,num)
     
    # Takes a partially filled-in grid and attempts to assign values to
    # all unassigned locations in such a way to meet the requirements
    # for Sudoku solution (non-duplication across rows, columns, and boxes)
    def solve_sudoku(self):
         
        # 'l' is a list variable that keeps the record of row and col in find_empty_location Function    
        l=[0,0]
         
        # If there is no unassigned location, we are done    
        if(not self.find_empty_location(l)):
            return True
         
        # Assigning list values to row and col that we got from the above Function 
        row=l[0]
        col=l[1]
         
        # consider digits 1 to 9
        for num in range(1,10):
             
            # if looks promising
            if(self.check_location_is_safe(row,col,num)):
                 
                # make tentative assignment
                self.arr[row][col]=num
     
                # return, if sucess, ya!
                if(self.solve_sudoku()):
                    return True
     
                # failure, unmake & try again
                self.arr[row][col] = 0
                 
        # this triggers backtracking        
        return False
 
# Driver main function to test above functions
"""if __name__=="__main__":
     
    # creating a 2D array for the grid
    grid=[[0 for x in range(9)]for y in range(9)]
     
    # assigning values to the grid
    grid=[[0,0,0,0,4,0,0,2,0],
          [0,5,0,9,0,0,0,0,0],
          [0,1,0,0,0,0,0,0,0],
          [0,0,0,8,0,0,1,0,5],
          [2,0,0,0,3,0,0,0,0],
          [0,0,0,0,0,0,9,0,0],
          [4,9,0,0,0,2,0,0,0],
          [3,0,0,0,0,0,0,6,0],
          [0,0,0,1,0,0,0,0,0]]
     
    # if sucess print the grid
    if(solve_sudoku(grid)):
        print_grid(grid)
    else:
        print "No solution exists"
"""
