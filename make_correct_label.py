import random

def check_markable_place(board):
    markable_place=[]
    for i in range(9):
        if board[i]==0:
            markable_place.append(i)
    
    return markable_place

def winner_exist(board):
    for i in range(3):
        if board[i]==board[i+3]==board[i+6]!=0:
            return True
        elif board[i*3]==board[i*3+1]==board[i*3+2]!=0:
            return True
    if board[0]==board[4]==board[8]!=0 or board[2]==board[4]==board[6]!=0:
        return True
    
    return False

def do_random_TicTacToe():
    while True:
        log={1:[],
            -1:[]}
        board=[0,0,0,  0,0,0,  0,0,0]

        turn=1  #1 == O   -1 == X
        while True:
            markable=check_markable_place(board)
            mark=random.choice(markable)

            log[turn].append((board[:],mark))

            board[mark]=turn
            if winner_exist(board):
                return log[turn]
            elif not 0 in board:
                break

            
            turn*=-1



def make_data(make_new_file=False,filename='data.txt',repeat=1):
    for w in range(repeat):
        if make_new_file:
            file=open(filename,'w')
        else:
            file=open(filename,'a')
        correct_label=do_random_TicTacToe()
        for i in correct_label:
            file.write(','.join(str(w) for w in i[0])+f'/{i[1]}\n')
        file.close()

        make_new_file=False
        print(f'{w+1}/{repeat}')



make_data(make_new_file=True,repeat=1000)
