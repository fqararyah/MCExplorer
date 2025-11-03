def canDistributeWithMaxOverflow(items, bags, maxOverflow):
    bag_index = 0
    current_weight = 0
    for item in items:
        if current_weight + item > bags[bag_index] + maxOverflow:
            # Move to the next bag
            bag_index += 1
            current_weight = item
            if bag_index >= len(bags):  # If no more bags are available
                return False
        else:
            current_weight += item
    
    return True