def canDistributeItems(items, bags, maxRelativeOverflow):
    bag_index = 0
    current_weight = 0
    current_bag_items = []
    bag_distributions = [[] for _ in bags]
    
    for item in items:
        # If adding the item would exceed the bag's relative capacity (max relative overflow)
        if current_weight + item > bags[bag_index] * (1 + maxRelativeOverflow):
            # Move to the next bag
            bag_distributions[bag_index] = current_bag_items
            bag_index += 1
            current_weight = item
            current_bag_items = [item]
            if bag_index >= len(bags):  # If no more bags are available
                return False, []
        else:
            current_weight += item  # Add item to the current bag
            current_bag_items.append(item)
    
    # Store the last set of items in the last bag
    bag_distributions[bag_index] = current_bag_items
    
    return True, bag_distributions

def minimize_relative_overflow_greedy(items, bags):
    left = 0  # Lower bound of relative overflow
    right = max(items) / min(bags)  # Upper bound of relative overflow (large overflows possible)
    best_max_relative_overflow = right
    best_distribution = None
    
    while left <= right:
        mid = (left + right) / 2  # Candidate for max relative overflow
        
        feasible, distribution = canDistributeItems(items, bags, mid)
        
        if feasible:
            best_max_relative_overflow = mid
            best_distribution = distribution
            right = mid - 0.01  # Try for a smaller relative overflow
        else:
            left = mid + 0.01  # Increase the allowed relative overflow
    
    return best_max_relative_overflow, best_distribution
