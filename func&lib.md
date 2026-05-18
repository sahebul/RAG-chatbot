### extend()
→ Used to add multiple items from another iterable into an existing list.
```py
fruits = ["apple", "banana"]
fruits.extend(["mango", "orange"])

print(fruits)
# ['apple', 'banana', 'mango', 'orange']
```
yield from
→ Used inside a generator to delegate iteration to another generator or iterable.

```py
def sub_generator():
    yield 1
    yield 2

def main_generator():
    yield from sub_generator()
    yield 3

for value in main_generator():
    print(value)
```
Output:
1
2
3