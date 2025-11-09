from math import copysign

def up(n, upper_bound):
        return int(n + copysign(1-upper_bound, n))
   
def round_if_in_range(n, lower_bound, upper_bound):
    if n < 0:
        return copysign(round_if_in_range(abs(n), lower_bound, upper_bound), -1)

    first_up_status = int(up(n%1, lower_bound))
    first_up = int(n) + first_up_status
    second_up_status = int(up(n%1, upper_bound))
    second_up = int(n) + second_up_status

    return ((first_up_status|second_up_status)&(first_up_status & second_up_status))*second_up + (first_up_status&(not second_up_status))*n + (int(not(first_up_status)) & int(not(second_up_status)))*first_up

#
#def down(n, lower_bound, upper_bound):
#    first_up_status = int(up(n%1, lower_bound))
#    first_up = int(n) + first_up_status
#    second_up_status = int(up(n%1, upper_bound))
#    second_up = int(n) + second_up_status
#    print(first_up_status)
#    print(first_up)
#    print(second_up_status)
#    print(second_up)
#    print(f"op 1: {((first_up_status|second_up_status)&(first_up_\
#status & second_up_status))}")
#    print(f"op 2: {(first_up_status&(~second_up_status))}")
#    print(f"op 3: {(~first_up_status & ~second_up_status)}")
#
#    return ((first_up_status|second_up_status)&(first_up_status &\
#second_up_status))*second_up + (first_up_status&(not second_up_statu\
#s))*n + (int(not(first_up_status)) & int(not(second_up_status)))*firs\
#t_up
