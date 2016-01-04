#!/usr/bin/env python


def zero_vector():
    return [0] * 4


def zero_matrix():
    return [zero_vector() for _ in xrange(4)]


def zero_tensor():
    return [zero_matrix() for _ in xrange(4)]


def parse_tensor(data):
    tensor = zero_tensor()
    lines = data.splitlines()
    for z in xrange(2):
        for y in xrange(4):
            line = lines[z * 5 + 1 + y]
            for x in xrange(4):
                if line[x] == '*':
                    tensor[z][y][x] = 1
    return tensor
    

J = parse_tensor("""
***.
*...
....
....

***.
*...
....
....

""")

Q = parse_tensor("""
**..
**..
....
....

**..
**..
....
....

""")

Z = parse_tensor("""
*...
*...
....
....

*...
***.
.**.
....

""")


def show_tensor(tensor):
    for y in xrange(4):
        for z in xrange(4):
            for x in xrange(4):
                value = tensor[z][y][x]
                print {
                    1: '*',
                    0: '.',
                    -1: '#'
                }[value],
            print ' ',
        print


def show_tensors(tensors):
    for tensor in tensors:
        show_tensor(tensor)
        print


def transform_tensor(tensor, function):
    transformed = zero_tensor()
    for z in xrange(4):
        for y in xrange(4):
            for x in xrange(4):
                transformed[z][y][x] = function(tensor, x, y, z)
    return transformed


def show_transformation(tensor, transformation, direction=1):
    show_tensors([
        tensor,
        transformation(tensor, direction=direction),
        apply_transformation(tensor, transformation, times=2, direction=direction),
        apply_transformation(tensor, transformation, times=3, direction=direction),
        apply_transformation(tensor, transformation, times=4, direction=direction),
    ])


def rotate_by_x(tensor, direction=1):
    return transform_tensor(tensor, lambda _, x, y, z: _[y][-z - 1][x])


def rotate_by_y(tensor, direction=1):
    if direction == 1:
        function = lambda _, x, y, z: _[x][y][-z - 1]
    else:
        function = lambda _, x, y, z: _[-x - 1][y][z]
    return transform_tensor(tensor, function)


def rotate_by_z(tensor, direction=1):
    return rotate_by_y(rotate_by_x(rotate_by_y(tensor, direction=-1)))


def project_tensor(tensor, function):
    projection = zero_matrix()
    for i in xrange(4):
        for j in xrange(4):
            projection[i][j] = function(tensor, i, j)
    return projection


def project_by_x(tensor, direction):
    return project_tensor(tensor, lambda _, z, y: tensor[z][y][0 if direction == 1 else -1])


def project_by_y(tensor, direction):
    return project_tensor(tensor, lambda _, z, x: tensor[z][0 if direction == 1 else -1][x])


def project_by_z(tensor, direction):
    return project_tensor(tensor, lambda _, y, x: tensor[0 if direction == 1 else -1][y][x])


def is_empty_matrix(matrix):
    for i in xrange(4):
        for j in xrange(4):
            if matrix[i][j]:
                return False
    return True


def shift_tensor(tensor, project, function):
    if is_empty_matrix(project(tensor)):
        return transform_tensor(tensor, function)
    return tensor


def shift_by_x(tensor, direction=1):
    return shift_tensor(
        tensor,
        lambda _: project_by_x(_, direction),
        lambda _, x, y, z: _[z][y][(x + direction) % 4]
    )


def shift_by_y(tensor, direction=1):
    return shift_tensor(
        tensor,
        lambda _: project_by_y(_, direction),
        lambda _, x, y, z: _[z][(y + direction) % 4][x]
    )


def shift_by_y(tensor, direction=1):
    return transform_tensor(tensor, lambda _, x, y, z: _[z][(y + 1) % 4][x])


def shift_by_z(tensor, direction=1):
    return shift_tensor(
        tensor,
        lambda _: project_by_z(_, direction),
        lambda _, x, y, z: _[(z + direction) % 4][y][x]
    )


def apply_transformation(tensor, transformation, direction=1, times=1):
    for _ in xrange(times):
        tensor = transformation(tensor, direction=direction)
    return tensor


def hash_tensor(tensor):
    hash = 0
    index = 0
    for z in xrange(4):
        for y in xrange(4):
            for x in xrange(4):
                index += 1
                hash += tensor[z][y][x] * 2 ** index
    return hash


def generate_permutations(tensor):
    for x_rotations in xrange(4):
        for y_rotations in xrange(4):
            for z_rotations in xrange(4):
                for x_shifts in xrange(3):
                    for x_direction in (-1, 1):
                        for y_shifts in xrange(3):
                            for y_direction in (-1, 1):
                                for z_shifts in xrange(3):
                                    for z_direction in (-1, 1):
                                        permutation = apply_transformation(tensor, rotate_by_x, times=x_rotations)
                                        permutation = apply_transformation(permutation, rotate_by_y, times=y_rotations)
                                        permutation = apply_transformation(permutation, rotate_by_z, times=z_rotations)
                                        permutation = apply_transformation(permutation, shift_by_x, direction=x_direction, times=x_shifts)
                                        permutation = apply_transformation(permutation, shift_by_y, direction=y_direction, times=y_shifts)
                                        permutation = apply_transformation(permutation, shift_by_z, direction=z_direction, times=z_shifts)
                                        yield permutation


def unique_tensors(tensors):
    hashes = set()
    for tensor in tensors:
        hash = hash_tensor(tensor)
        if hash not in hashes:
            yield tensor
        hashes.add(hash)


def union_tensor_hashes(a, b):
    return a | b


def tensor_hashes_intersect(a, b):
    return a & b
 

def solve_(path, done, hashes, todo):
    total = len(hashes)
    for index, hash in enumerate(hashes):
        if not path:
            print index, '/', total
        if not tensor_hashes_intersect(done, hash):
            if todo:
                for solution in solve_(path + [hash], union_tensor_hashes(done, hash), todo[0], todo[1:]):
                    yield solution
            else:
                yield path + [hash]


def solve(Zs, Js, Qs):
    Z_hashes = [hash_tensor(_) for _ in Zs]
    J_hashes = [hash_tensor(_) for _ in Js]
    Q_hashes = [hash_tensor(_) for _ in Qs]
    done = hash_tensor(zero_tensor())
    todo = [Z_hashes] * 3 + [J_hashes] * 3 + [Q_hashes] * 2
    for solution in solve_([], done, todo[0], todo[1:]):
        yield [unhash_tensor(_) for _ in solution]
 

def unhash_tensor(hash):
    tensor = zero_tensor()
    index = 0
    for z in xrange(4):
        for y in xrange(4):
            for x in xrange(4):
                index += 1
                if hash & 2 ** index:
                    tensor[z][y][x] = 1
    return tensor
    

def dump_solutions(solutions):
    with open('solutions.json', 'w') as file:
        file.write(repr(solutions))

        
def load_solutions():
    with open('solutions.json') as file:
        return eval(file.read())


def add_tensors(a, b):
    sum = zero_tensor()
    for z in xrange(4):
        for y in xrange(4):
            for x in xrange(4):
                sum[z][y][x] = a[z][y][x] + b[z][y][x]
    return sum


def scale_tensor(tensor, scale):
    scaled = zero_tensor()
    for z in xrange(4):
        for y in xrange(4):
            for x in xrange(4):
                scaled[z][y][x] = tensor[z][y][x] * scale
    return scaled


def solution_tensor(solution):
    Z1, Z2, Z3, J1, J2, J3, Q1, Q2 = solution
    tensor = zero_tensor()
    tensor = add_tensors(tensor, scale_tensor(Z1, 1))
    tensor = add_tensors(tensor, scale_tensor(Z2, 1))
    tensor = add_tensors(tensor, scale_tensor(Z3, 1))
    tensor = add_tensors(tensor, scale_tensor(J1, -1))
    tensor = add_tensors(tensor, scale_tensor(J2, -1))
    tensor = add_tensors(tensor, scale_tensor(J3, -1))
    # Qs are multiplied by 0
    return tensor
