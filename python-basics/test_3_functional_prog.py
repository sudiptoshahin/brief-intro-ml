items = [1, 2, 3, 4, 5]


def square_items(items):
    squared = []
    for val in items:
        squared.append(val**2)
    return squared


def square_single_item(item):
    return item**2


# temp_squared = square_items(items)
temp_squared = list(map(square_single_item, items))
# temp_squared = list(map(lambda x: x**2, items))

print(f"Before: {items}")
print(f'After: {temp_squared}')

num_list = list(range(-5, 5))
less_than_zero = list(filter(lambda x: x < 0, num_list))
print('List: ', num_list)
print(f'Less than zero list: {less_than_zero}')
