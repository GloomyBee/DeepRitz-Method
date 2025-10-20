# 定义一个判断年龄是否成年的函数
def is_adult(age):
    return age >= 18

# 定义一个函数，它接受一个列表和一个判断函数作为参数
def filter_adult(people, filter_func):
    result = []
    for person in people:
        # 在这里，我们调用了传入的 filter_func 函数
        if filter_func(person['age']):
            result.append(person)
    return result

# 我们的测试数据
people_list = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 17},
    {'name': 'Charlie', 'age': 30}
]

# 使用 filter_adult 函数，并将 is_adult 函数作为参数传进去
adults = filter_adult(people_list, is_adult)

print(adults)
# 输出: [{'name': 'Alice', 'age': 25}, {'name': 'Charlie', 'age': 30}]
