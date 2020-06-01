def get_open(page):
    return 1 + (page//2)
def solve(n, p):
    return min(get_open(p) - get_open(1), get_open(n) - get_open(p))
n = int(input().strip())
p = int(input().strip())
result = solve(n, p)
print(result)
EOF
def angryChildren(k, arr):
    arr = sorted(arr)
    res = arr[-1]
    for ind in range(len(arr)-k+1):
        res = min(res, arr[ind+k-1] - arr[ind])
    return res
if __name__ == "__main__":
    n = int(input().strip())
    k = int(input().strip())
    arr = []
    arr_i = 0
    for arr_i in range(n):
        arr_t = int(input().strip())
        arr.append(arr_t)
    result = angryChildren(k, arr)
    print(result)
EOF
def countingValleys(n, s):
    res = 0
    in_valley = 0
    curr = 0
    for step in s:
        if step == 'U':
            curr += 1
        else:
            curr -= 1
        if curr < 0 and in_valley == 0:
            in_valley = 1
        if in_valley == 1 and curr == 0:
            in_valley = 0
            res += 1
    return res
if __name__ == "__main__":
    n = int(input().strip())
    s = input().strip()
    result = countingValleys(n, s)
    print(result)
EOF
def validate(string):
    for ind in range(len(string)-1):
        if string[ind] == string[ind + 1]:
            return False
    return True
def twoCharaters(string):
    str_set = set(list(string))
    variants = combinations(str_set, 2)
    max_res = 0
    for comb in variants:
        t = [c for c in string if c == comb[0] or c == comb[1]]
        if validate(t):
            max_res = max(max_res, len(t))
    return max_res
if __name__ == "__main__":
    l = int(input().strip())
    s = input().strip()
    result = twoCharaters(s)
    print(result)
EOF
def beautifulTriplets(d, arr):
    res = 0
    for el in arr:
        if el + d in arr and el + 2*d in arr:
            res += 1
    return res
if __name__ == "__main__":
    n, d = input().strip().split(' ')
    n, d = [int(n), int(d)]
    arr = list(map(int, input().strip().split(' ')))
    result = beautifulTriplets(d, arr)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TopVotedCandidate:
    def __init__(self, persons, times):
        self.times = times
        self.leader = []                        
        counts = defaultdict(int)               
        max_count = 0                           
        for person, time in zip(persons, times):
            counts[person] += 1
            if counts[person] > max_count:      
                max_count += 1
                leaders = [person]
            elif counts[person] == max_count:   
                leaders.append(person)
            self.leader.append(leaders[-1])     
    def q(self, t):
        i = bisect.bisect_left(self.times, t)
        if i == len(self.times) or self.times[i] > t:
            return self.leader[i - 1]
        return self.leader[i]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def beautifulArray(self, N):
        if N == 1:
            return [1]
        evens = self.beautifulArray(N // 2)
        if N % 2 == 0:          
            odds = evens[:]
        else:
            odds = self.beautifulArray((N + 1) // 2)
        odds = [(2 * i) - 1 for i in odds]
        evens = [2 * i for i in evens]
        return evens + odds
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rotatedDigits(self, N):
        count = 0
        bad = {"3", "4", "7"}
        opposites = {"2", "5", "6", "9"}
        for i in range(1, N + 1):
            digits = set(str(i))
            if not bad.intersection(digits) and opposites.intersection(digits):
                count += 1
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def addBinary(self, a, b):
        result = []
        carry = 0
        i = len(a)-1
        j = len(b)-1
        while carry or i >= 0 or j >= 0:
            total = carry
            if i >= 0:
                total += int(a[i])
                i -= 1
            if j >= 0:
                total += int(b[j])
                j -= 1
            result.append(str(total % 2))
            carry = total // 2
        return "".join(result[::-1])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
class Solution(object):
    def maxPoints(self, points):
        if len(points) <= 2:
            return len(points)
        overall_max = 2
        for i, point in enumerate(points):  
            gradients = defaultdict(int)    
            max_points = 1                  
            for point_2 in points[i+1:]:    
                if point.x == point_2.x:
                    if point.y == point_2.y:    
                        max_points += 1
                    else:                       
                        gradients['inf'] += 1
                else:
                    gradient = (point_2.y - point.y) / float(point_2.x - point.x)
                    gradients[gradient] += 1
            if gradients:
                max_points += max(gradients.values())
            overall_max = max(overall_max, max_points)
        return overall_max
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def oddEvenList(self, head):
        even_head = even = ListNode(None)
        odd_head = odd = ListNode(None)
        while head:
            odd.next = head         
            odd = odd.next          
            even.next = head.next   
            even = even.next        
            head = head.next.next if even else None 
        odd.next = even_head.next   
        return odd_head.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def pathSum(self, root, sum):
        paths = defaultdict(int)    
        paths[0] = 1                
        def helper(node, partial):
            if not node:
                return 0
            partial += node.val
            count = paths[partial - sum]    
            paths[partial] += 1
            count += helper(node.left, partial)
            count += helper(node.right, partial)
            paths[partial] -= 1
            return count
        return helper(root, 0)
EOF
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        line = input().split()
        name, scores = line[0], line[1:]
        scores = list(map(float, scores))
        student_marks[name] = scores
    query_name = input()
    student_score = student_marks[query_name]
    print("%.2f" %(sum(student_score)/len(student_score)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def splitLoopedString(self, strs):
        result = None
        best = [max(s, s[::-1]) for s in strs]
        for i, s in enumerate(strs):
            t = s[::-1]
            for j in range(len(s)):
                test = s[j:] + "".join(best[i + 1:] + best[:i]) + s[:j]
                test2 = t[j:] + "".join(best[i + 1:] + best[:i]) + t[:j]
                result = max(result, test, test2)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        if m == 0:
            return 0
        result = 0
        bit = int(log(n, 2))        
        while bit >= 0 and ((m  & (1 << bit)) == (n  & (1 << bit))):
            if (m  & (1 << bit)):   
                result += 2**bit
            bit -= 1
        return result
EOF
def solveMeFirst(a ,b):
    return a + b
res = solveMeFirst(int(input()) ,int(input()))
print(res)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def distanceK(self, root, target, K):
        results = []
        def nodes_at_distance(node, distance):      
            if not node:
                return
            if distance == 0:
                results.append(node.val)
            else:
                nodes_at_distance(node.left, distance - 1)
                nodes_at_distance(node.right, distance - 1)
        def helper(node):                           
            if not node:
                return -1
            if node == target:
                nodes_at_distance(node, K)          
                return 0
            left, right = helper(node.left), helper(node.right)
            if left == -1 and right == -1:          
                return -1
            distance_to_target = 1 + max(left, right)   
            if K - distance_to_target == 0:         
                nodes_at_distance(node, 0)
            elif K - distance_to_target > 0:        
                other_side = node.left if left == -1 else node.right
                nodes_at_distance(other_side, K - distance_to_target - 1)
            return distance_to_target
        helper(root)
        return results
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minSwap(self, A, B):
        prev_no_swap, prev_swap = 0, 1              
        for i in range(1, len(A)):                  
            no_swap, swap = float("inf"), float("inf")  
            if A[i] > A[i - 1] and B[i] > B[i - 1]:
                no_swap = prev_no_swap
                swap = 1 + prev_swap
            if A[i] > B[i - 1] and B[i] > A[i - 1]:
                no_swap = min(no_swap, prev_swap)
                swap = min(swap, 1 + prev_no_swap)
            prev_no_swap, prev_swap = no_swap, swap
        return min(prev_no_swap, prev_swap)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def postorder(self, root):
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node.val)
            for child in node.children:
                stack.append(child)
        return result[::-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def candyCrush(self, board):
        rows, cols = len(board), len(board[0])
        while True:
            stable = True
            to_crush = [[False for _ in range(cols)] for _ in range(rows)]      
            for c in range(cols):
                for r in range(rows):
                    if r < rows - 2 and board[r][c] == board[r + 1][c] == board[r + 2][c] and board[r][c] != 0:
                        to_crush[r][c] = to_crush[r + 1][c] = to_crush[r + 2][c] = True
                        stable = False
                    if c < cols - 2 and board[r][c] == board[r][c + 1] == board[r][c + 2] and board[r][c] != 0:
                        to_crush[r][c] = to_crush[r][c + 1] = to_crush[r][c + 2] = True
                        stable = False
            if stable:  
                return board
            for c in range(cols):
                new_col = [0 for _ in range(rows)]
                new_r = rows - 1
                for r in range(rows - 1, -1, -1):       
                    if not to_crush[r][c]:
                        new_col[new_r] = board[r][c]    
                        new_r -= 1                      
                if new_r != -1:                         
                    for r in range(rows):
                        board[r][c] = new_col[r]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sortArray(self, nums):
        return sorted(nums)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findTheDifference(self, s, t):
        counts = [0 for _ in range(26)]
        for c in s:
            counts[ord(c) - ord("a")] += 1
        for c in t:
            index = ord(c) - ord("a")
            counts[index] -= 1
            if counts[index] < 0:
                return c
EOF
if __name__ == '__main__':
    N = int(input())
    names = []
    for N_itr in range(N):
        firstNameEmailID = input().split()
        firstName = firstNameEmailID[0]
        emailID = firstNameEmailID[1]
        if re.search(r"            names.append(firstName)
print(*sorted(names),sep='\n')
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLongestWord(self, s, d):
        def is_subsequence(s, t):  
            i, j = 0, 0
            while i < len(s) and (len(t) - j) >= (len(s) - i):  
                if s[i] == t[j]:
                    i += 1
                j += 1
            if i == len(s):
                return True
            return False
        d.sort(key=lambda x: (-len(x), x))      
        for word in d:
            if is_subsequence(word, s):
                return word
        return ""
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def shortestBridge(self, A):
        rows, cols = len(A), len(A[0])
        visited = set()
        perimeter = set()
        def neighbours(r, c):       
            if r != 0:
                yield (r - 1, c)
            if r != rows - 1:
                yield (r + 1, c)
            if c != 0:
                yield (r, c - 1)
            if c != rows - 1:
                yield (r, c + 1)
        def get_perimeter(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return
            if A[r][c] == 0 or (r, c) in visited:
                return
            visited.add((r, c))
            for r1, c1 in neighbours(r, c):
                if A[r1][c1] == 0:      
                    perimeter.add((r1, c1))
                else:                   
                    get_perimeter(r1, c1)
        for r in range(rows):
            for c in range(cols):
                if perimeter:           
                    break
                get_perimeter(r, c)
        steps = 1                       
        while True:
            new_perimeter = set()
            for r, c in perimeter:
                for r1, c1 in neighbours(r, c):
                    if (r1, c1) in visited:
                        continue
                    if A[r1][c1] == 1:  
                        return steps
                    new_perimeter.add((r1, c1))
            visited |= new_perimeter
            perimeter = new_perimeter
            steps += 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def preorderTraversal(self, root):
        if not root:
            return []
        preorder = []
        stack = [root]
        while stack:
            node = stack.pop()
            preorder.append(node.val)
            if node.right:
                stack.append(node.right)    
            if node.left:
                stack.append(node.left)
        return preorder
class Solution2(object):
    def preorderTraversal(self, root):
        result = []
        self.preorder(root, result)
        return result
    def preorder(self, node, result):
        if not node:
            return
        result.append(node.val)
        self.preorder(node.left, result)
        self.preorder(node.right, result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isSubsequence(self, s, t):
        if not s:
            return True
        i = 0           
        for c in t:
            if c == s[i]:
                i += 1
                if i == len(s):
                    return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def combine(self, n, k):
        if k == 0:          
            return [[]]
        if n < k:           
            return []
        without_last = self.combine(n-1, k)
        with_last = [[n] + combo for combo in self.combine(n-1, k-1)]
        return with_last + without_last
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def pathSum(self, root, sum):
        paths = []
        self.preorder(root, sum, [], paths)
        return paths
    def preorder(self, node, target, partial, paths):
        if not node:
            return
        target -= node.val
        partial.append(node.val)
        if target == 0 and not node.left and not node.right:
            paths.append(partial[:])
        self.preorder(node.left, target, partial, paths)
        self.preorder(node.right, target, partial, paths)
        partial.pop()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minSwapsCouples(self, row):
        n = len(row) // 2   
        couple_to_location = [[] for _ in range(n)]     
        for i, person in enumerate(row):                
            couple_to_location[person // 2].append(i // 2)
        print(couple_to_location)
        adjacency = [[] for _ in range(n)]              
        for a, b in couple_to_location:                 
            adjacency[a].append(b)
            adjacency[b].append(a)
        print(adjacency)
        swaps = n                                       
        for start in range(n):
            if not adjacency[start]:
                continue
            swaps -= 1                                  
            a = start                                   
            b = adjacency[start].pop()                  
            while b != start:
                adjacency[b].remove(a)                  
                a, b = b, adjacency[b].pop()            
        return swaps
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def increasingTriplet(self, nums):
        smallest, next_smallest = float("inf"), float("inf")
        for num in nums:
            smallest = min(smallest, num)
            if num > smallest:
                next_smallest = min(next_smallest, num)
            if num > next_smallest:
                return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def smallestRangeI(self, A, K):
        range = max(A) - min(A)
        return max(0, range - 2 * K)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyLinkedList(object):
    def __init__(self):
        self.list = []
    def get(self, index):
        if index >= len(self.list):
            return -1
        return self.list[index]
    def addAtHead(self, val):
        self.list = [val] + self.list
    def addAtTail(self, val):
        self.list.append(val)
    def addAtIndex(self, index, val):
        if index > len(self.list):
            return
        self.list.insert(index, val)
    def deleteAtIndex(self, index):
        if index >= len(self.list):
            return
        del self.list[index]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findDisappearedNumbers(self, nums):
        for num in nums:
            num = abs(num)                          
            nums[num - 1] = -abs(nums[num - 1])     
        return [i + 1 for i, num in enumerate(nums) if num > 0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class PhoneDirectory(object):
    def __init__(self, maxNumbers):
        self.free = set(range(maxNumbers))
    def get(self):
        return self.free.pop() if self.free else -1
    def check(self, number):
        return number in self.free
    def release(self, number):
        self.free.add(number)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Excel(object):
    def _indices(self, r, c):  
        return [r - 1, ord(c) - ord("A")]
    def __init__(self, H, W):
        rows, cols = self._indices(H, W)
        self.excel = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
    def set(self, r, c, v):
        r, c, = self._indices(r, c)
        self.excel[r][c] = v
    def get(self, r, c):
        r, c = self._indices(r, c)
        return self.get_i(r, c)  
    def get_i(self, r, c):      
        contents = self.excel[r][c]
        if isinstance(contents, int):  
            return contents
        total = 0
        for cells in contents:
            cell_range = cells.split(":")
            r1, c1 = self._indices(int(cell_range[0][1:]), cell_range[0][0])
            if len(cell_range) == 1:  
                r2, c2 = r1, c1
            else:  
                r2, c2 = self._indices(int(cell_range[1][1:]), cell_range[1][0])
            for row in range(r1, r2 + 1):
                for col in range(c1, c2 + 1):
                    total += self.get_i(row, col)  
        return total
    def sum(self, r, c, strs):
        r, c = self._indices(r, c)
        self.excel[r][c] = strs
        return self.get_i(r, c)  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MedianFinder:
    def __init__(self):
        self.lower = []     
        self.higher = []    
    def addNum(self, num):
        if not self.lower or num <= -self.lower[0]:     
            heapq.heappush(self.lower, -num)
        else:
            heapq.heappush(self.higher, num)
        if len(self.higher) > len(self.lower):          
            heapq.heappush(self.lower, -heapq.heappop(self.higher))
        elif len(self.lower) > 1 + len(self.higher):    
            heapq.heappush(self.higher, -heapq.heappop(self.lower))
    def findMedian(self):
        if len(self.lower) > len(self.higher):
            return float(-self.lower[0])
        return (-self.lower[0] + self.higher[0]) / 2.0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isRectangleCover(self, rectangles):
        min_r, min_c = float("inf"), float("inf")       
        max_r, max_c = float("-inf"), float("-inf")
        area = 0                                        
        corners = defaultdict(int)                      
        for r1, c1, r2, c2 in rectangles:
            area += (r2 - r1) * (c2 - c1)
            min_r = min(min_r, r1)
            min_c = min(min_c, c1)
            max_r = max(max_r, r2)
            max_c = max(max_c, c2)
            corners[(r1, c1)] += 1
            corners[(r2, c2)] += 1
            corners[(r1, c2)] += 1
            corners[(r2, c1)] += 1
        rows = max_r - min_r
        cols = max_c - min_c
        if area != rows * cols:
            return False
        for r, c in corners:
            if r in {min_r, max_r} and c in {min_c, max_c}:
                if corners[(r, c)] != 1:
                    return False
            elif corners[(r, c)] % 2 != 0:
                return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def newInteger(self, n):
        result = []
        while n:
            result.append(str(n % 9))
            n //= 9
        return int("".join(result[::-1]))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def anagramMappings(self, A, B):
        B_to_int = {}
        for i, b in enumerate(B):
            B_to_int[b] = i
        result = []
        for a in A:
            result.append(B_to_int[a])
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getFactors(self, n):
        return self.factorise(n, 2, [], [])
    def factorise(self, n, trial, partial, factors):
        while trial * trial <= n:       
            if n % trial == 0:                                  
                factors.append(partial + [n//trial, trial])     
                self.factorise(n//trial, trial, partial + [trial], factors) 
            trial += 1
        return factors
class Solution2(object):
    def getFactors(self, n):
        stack = [(n, 2, [])]        
        factors = []
        while stack:
            num, trial, partial = stack.pop()
            while trial * trial <= num:     
                if num % trial == 0:
                    factors.append(partial + [num//trial, trial])
                    stack.append((num//trial, trial, partial + [trial]))
                trial += 1
        return factors
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def largestBSTSubtree(self, root):
        self.largest = 0
        def is_bst(node):   
            if not node:
                return float("-inf"), float("inf"), 0
            left_bst = is_bst(node.left)
            right_bst = is_bst(node.right)
            if left_bst[2] != -1 and right_bst[2] != -1:        
                if left_bst[0] < node.val < right_bst[1]:       
                    size = 1 + left_bst[2] + right_bst[2]       
                    self.largest = max(self.largest, size)
                    return max(right_bst[0], node.val), min(left_bst[1], node.val), size
            return 0, 0, -1
        is_bst(root)    
        return self.largest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def sortedSquares(self, A):
        left, right = 0, len(A) - 1
        result = []
        while left <= right:
            if abs(A[left]) > abs(A[right]):
                result.append(A[left] * A[left])
                left += 1
            else:
                result.append(A[right] * A[right])
                right -= 1
        return result[::-1]     
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def uncommonFromSentences(self, A, B):
        counts = Counter(A.split(" ")) + Counter(B.split(" "))
        return [word for word, count in counts.items() if count == 1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def moveZeroes(self, nums):
        i = 0       
        for num in nums:
            if num != 0:
                nums[i] = num
                i += 1
        nums[i:] = [0] * (len(nums) - i)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMinArrowShots(self, points):
        arrows, last_arrow = 0, float("-inf")
        points.sort(key = lambda x: x[1])
        for start, end in points:
            if start > last_arrow:
                arrows += 1
                last_arrow = end
        return arrows
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestDistance(self, grid):
        rows, cols = len(grid), len(grid[0])
        house = 0                       
        distances = deepcopy(grid)      
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != 1:     
                    continue
                q = [(row, col)]            
                house_dist = 1
                while q:
                    new_q = []
                    for r, c in q:
                        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            if 0 <= r + dr < rows and 0 <= c + dc < cols and grid[r + dr][c + dc] == -house:
                                grid[r + dr][c + dc] -= 1           
                                new_q.append((r + dr, c + dc))      
                                distances[r + dr][c + dc] += house_dist     
                    house_dist += 1
                    q = new_q
                house += 1
        reachable = [distances[r][c] for r in range(rows) for c in range(cols) if grid[r][c] == -house]
        return -1 if not reachable else min(reachable)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]:
            return False
        rows, cols = len(matrix), len(matrix[0])
        r, c = 0, cols-1
        while r < rows and c >= 0:
            if matrix[r][c] == target:
                return True
            if target > matrix[r][c]:
                r += 1
            else:
                c -= 1
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maximalSquare(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        rows, cols = len(matrix), len(matrix[0])
        max_side = 0
        square_sides = [0] * cols       
        for r in range(rows):
            new_square_sides = [int(matrix[r][0])] + [0 for _ in range(cols-1)]
            for c in range(1, cols):
                if matrix[r][c] == '1':
                    new_square_sides[c] = 1 + min(new_square_sides[c-1], square_sides[c], square_sides[c-1])
            max_side = max(max_side, max(new_square_sides))
            square_sides = new_square_sides
        return max_side**2
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def increasingBST(self, root, tail=None):       
        if root is None:
            return tail
        copy_root = TreeNode(root.val)              
        copy_root.right = self.increasingBST(root.right, tail)
        return self.increasingBST(root.left, copy_root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canIWin(self, maxChoosableInteger, desiredTotal):
        if maxChoosableInteger * (maxChoosableInteger + 1) // 2 < desiredTotal:
            return False  
        return self.next_player_win(desiredTotal, list(range(1, maxChoosableInteger + 1)), {})
    def next_player_win(self, target, unused, memo):
        if unused[-1] >= target:
            return True
        tup = tuple(unused)
        if tup in memo:
            return memo[tup]
        for i in range(len(unused) - 1, -1, -1):
            opposition_win = self.next_player_win(target - unused[i], unused[:i] + unused[i + 1:], memo)
            if not opposition_win:
                memo[tup] = True
                return True
        memo[tup] = False
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def countUnivalSubtrees(self, root):
        self.univariates = 0
        self.is_univariate(root)
        return self.univariates
    def is_univariate(self, root):
        if not root:
            return True
        left_uni = self.is_univariate(root.left)
        right_uni = self.is_univariate(root.right)
        if left_uni and right_uni:
            if (not root.left or root.left.val == root.val) and (not root.right or root.right.val == root.val):
                self.univariates += 1
                return True
        return False
class Solution2(object):
    def countUnivalSubtrees(self, root):
        self.univariates = 0
        self.preorder(root)
        return self.univariates
    def preorder(self, root):
        if not root:
            return
        if self.is_univariate(root):
            self.univariates += 1
        self.preorder(root.left)
        self.preorder(root.right)
    def is_univariate(self, root):
        if not root:
            return True
        if root.left and root.left.val != root.val:
            return False
        if root.right and root.right.val != root.val:
            return False
        return self.is_univariate(root.left) and self.is_univariate(root.right)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def removeComments(self, source):
        removed = []
        comment_block = False
        new_line = []
        for line in source:
            i = 0
            while i < len(line):
                test = line[i:i + 2]
                if not comment_block and test == "/*":      
                    comment_block = True                    
                    i += 2
                elif not comment_block and test == "//":    
                    i = len(line)
                elif comment_block and test == "*/":        
                    comment_block = False
                    i += 2
                elif comment_block:                         
                    i += 1
                else:                                       
                    new_line.append(line[i])
                    i += 1
            if not comment_block and new_line:              
                removed.append("".join(new_line))
                new_line = []
        if new_line:                                        
            removed.append("".join(new_line))
        return removed
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def preorder(self, root):
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node.val)
            for child in reversed(node.children):
                stack.append(child)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def threeEqualParts(self, A):
        one_count = sum(A)
        if one_count == 0:              
            return [0, 2]
        ones_per_part, remainder = divmod(one_count, 3)
        if remainder != 0:              
            return [-1, -1]
        first_start = 0
        while A[first_start] == 0:      
            first_start += 1
        first_end = first_start
        count = 1
        while count < ones_per_part:    
            first_end += 1
            count += A[first_end]
        length = first_end - first_start + 1
        second_start = first_end + 1
        while A[second_start] == 0:     
            second_start += 1
        if A[first_start:first_end + 1] != A[second_start:second_start + length]:
            return [-1, -1]             
        third_start = second_start + length
        while A[third_start] == 0:
            third_start += 1
        if A[first_start:first_end + 1] != A[third_start:third_start + length]:
            return [-1, -1]
        trailing_zeros = len(A) - third_start - length  
        first_end += trailing_zeros
        second_end = second_start + length - 1 + trailing_zeros
        if second_start < first_end + 1 or third_start < second_end + 1:    
            return [-1, -1]
        return [first_end, second_end + 1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def intervalIntersection(self, A, B):
        result = []
        i, j = 0, 0                         
        while i < len(A) and j < len(B):
            last_start = max(A[i].start, B[j].start)
            first_end = min(A[i].end, B[j].end)
            if last_start <= first_end:     
                result.append(Interval(s=last_start, e=first_end))
            if A[i].end < B[j].end:         
                i += 1
            else:
                j += 1
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MagicDictionary(object):
    def __init__(self):
        self.root = {}      
    def buildDict(self, dict):
        for word in dict:
            node = self.root
            for c in word:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node["
    def search(self, word):
        def helper(i, mismatches, node):    
            if mismatches == 2:             
                return False
            if i == len(word):              
                return "
            for c in node.keys():
                if c == "
                    continue
                if helper(i + 1, mismatches + (c != word[i]), node[c]):
                    return True
            return False
        return helper(0, 0, self.root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countSubstrings(self, s):
        count = 0
        for i in range(2 * len(s) + 1):
            left = right = i // 2
            if i % 2 == 1:
                right += 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def subtreeWithAllDeepest(self, root):
        Result = namedtuple("Result", ["node", "depth"])    
        def helper(node):
            if not node:
                return Result(None, -1)
            left_result, right_result = helper(node.left), helper(node.right)
            depth_diff = left_result.depth - right_result.depth
            if depth_diff == 0:
                return Result(node, left_result.depth + 1)
            if depth_diff > 0:
                return Result(left_result.node, left_result.depth + 1)
            return Result(right_result.node, right_result.depth + 1)
        return helper(root).node
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def customSortString(self, S, T):
        result = []
        t_count = Counter(T)
        for c in S:
            result += [c] * t_count[c]      
            del t_count[c]
        for c, count in t_count.items():
            result += [c] * count
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def strangePrinter(self, s):
        s = "".join([a for a, b in zip(s, "
        memo = {}
        def helper(i, j):
            if j - i + 1 <= 1:                  
                return j - i + 1
            if (i, j) in memo:
                return memo[(i, j)]
            min_prints = 1 + helper(i + 1, j)   
            for k in range(i + 1, j + 1):
                if s[k] == s[i]:                
                    min_prints = min(min_prints, helper(i, k - 1) + helper(k + 1, j))
            memo[(i, j)] = min_prints
            return min_prints
        return helper(0, len(s) - 1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def profitableSchemes(self, G, P, group, profit):
        MOD = 10 ** 9 + 7
        schemes = [[0] * (G + 1) for _ in range(P + 1)]     
        schemes[0][0] = 1
        for job_profit, job_gang in zip(profit, group):
            for p in range(P, -1, -1):                      
                for g in range(G, job_gang - 1, -1):        
                    capped_profit = min(P, p + job_profit)  
                    schemes[capped_profit][g] += schemes[p][g - job_gang]   
        return sum(schemes[-1]) % MOD                       
EOF
n = int(input().strip())
a = list(map(int, input().strip().split(' ')))
totalSwap = 0
for i in range(len(a)-1):
    flag = False
    for j in range(len(a)-i-1):
        if a[j] > a[j+1]:
            a[j+1], a[j] = a[j], a[j+1]
            totalSwap += 1
            flag = True
    if flag == False:
        break
print("Array is sorted in "+str(totalSwap)+" swaps.")
print("First Element:",a[0])
print("Last Element:",a[-1])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestValues(self, root):
        result = []
        if not root:
            return []
        queue = [root]
        while queue:
            new_queue = []
            max_val = float("-inf")
            for node in queue:
                max_val = max(max_val, node.val)
                if node.left:
                    new_queue.append(node.left)
                if node.right:
                    new_queue.append(node.right)
            result.append(max_val)
            queue = new_queue
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def deleteNode(self, root, key):
        if not root:
            return None
        if key > root.val:  
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if not (root.left and root.right):  
                root = root.left or root.right
            else:  
                next_largest = root.right
                while next_largest.left:
                    next_largest = next_largest.left
                root.val = next_largest.val
                root.right = self.deleteNode(root.right, root.val)
        return root
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMin(self, nums):
        left = 0
        right = len(nums)-1
        while left < right:
            if nums[left] <= nums[right]:   
                break
            mid = (left + right) // 2
            if nums[right] < nums[mid]:     
                left = mid + 1
            else:                           
                right = mid                 
        return nums[left]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def leastOpsExpressTarget(self, x, target):
        pos = neg = powers = 0
        while target:
            target, remainder = divmod(target, x)       
            if powers == 0:
                pos, neg = remainder * 2, (x - remainder) * 2   
            else:
                pos, neg = min(remainder * powers + pos, (remainder + 1) * powers + neg), \
                           min((x - remainder) * powers + pos, (x - remainder - 1) * powers + neg)
            powers += 1
        return min(pos, powers + neg) - 1   
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def removeDuplicates(self, nums):
        next_new = 0        
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i - 1]:
                nums[next_new] = nums[i]
                next_new += 1
        return next_new
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def matrixReshape(self, nums, r, c):
        rows, cols = len(nums), len(nums[0])
        if rows * cols != r * c:            
            return nums
        reshaped = [[]]
        for i in range(rows):
            for j in range(cols):
                if len(reshaped[-1]) == c:  
                    reshaped.append([])
                reshaped[-1].append(nums[i][j])
        return reshaped
EOF
n = int(input().strip())
for i in range(1, 11):
    print(str(n) +" x " + str(i) + " = " + str(n *i))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def kClosest(self, points, K):
        return heapq.nsmallest(K, points, lambda x, y: x * x + y * y)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def pyramidTransition(self, bottom, allowed):
        triangles = defaultdict(list)
        for triple in allowed:
            triangles[triple[:-1]].append(triple[-1])
        def helper(prev, current):          
            if len(prev) == 1:              
                return True
            n = len(current)
            if n == len(prev) - 1:          
                return helper(current, "")
            colours = triangles[prev[n:n + 2]]  
            if not colours:
                return False
            for colour in colours:
                if helper(prev, current + colour):
                    return True
            return False
        return helper(bottom, "")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findPoisonedDuration(self, timeSeries, duration):
        poisoned = 0
        timeSeries.append(float("inf"))
        for i in range(1, len(timeSeries)):
            poisoned += min(duration, timeSeries[i] - timeSeries[i- 1])
        return poisoned
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def flipEquiv(self, root1, root2):
        if not root1 and not root2:     
            return True
        if not root1 or not root2:      
            return False
        if root1.val != root2.val:
            return False
        return (self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right)) \
               or (self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def construct(self, grid):
        def helper(r, c, side):     
            if side == 1:           
                return Node(bool(grid[r][c]), True, None, None, None, None)
            top_left = helper(r, c, side // 2)
            top_right = helper(r, c + side // 2, side // 2)
            bottom_left = helper(r + side // 2, c, side // 2)
            bottom_right = helper(r + side // 2, c + side // 2, side // 2)
            if top_left.isLeaf and top_right.isLeaf and bottom_left.isLeaf and bottom_right.isLeaf:
                if top_left.val == top_right.val == bottom_left.val == bottom_right.val:
                    return Node(top_left.val, True, None, None, None, None)
            node_val = any((top_left.val, top_right.val, bottom_left.val, bottom_right.val))
            return Node(node_val, False, top_left, top_right, bottom_left, bottom_right)
        if not grid:
            return None
        return helper(0, 0, len(grid))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def hasAlternatingBits(self, n):
        n, bit = divmod(n, 2)       
        while n:
            if n % 2 == bit:        
                return False
            n, bit = divmod(n, 2)
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def merge(self, intervals):
        intervals.sort(key=lambda x : x.start)
        merged = []
        for interval in intervals:
            if not merged or merged[-1].end < interval.start:
                merged.append(interval)
            else:
                merged[-1].end = max(merged[-1].end, interval.end)
        return merged
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxArea(self, height):
        left = 0
        right = len(height)-1
        max_area = (right - left) * min(height[right], height[left])
        while left < right:
            if height[left] < height[right]:    
                left += 1                       
            else:
                right -= 1
            max_area = max(max_area, (right - left) * min(height[right], height[left]))
        return max_area
EOF
def solve(n, bar, d, m):
    res = 0
    for a_i in range(len(bar)-m+1):
        if sum(bar[a_i:a_i+m]) == d:
            res += 1
    return res
n = int(input().strip())
bar = list(map(int, input().strip().split(' ')))
d, m = input().strip().split(' ')
d, m = [int(d), int(m)]
result = solve(n, bar, d, m)
print(result)
EOF
def calc_pattern():
    max_n = 50
    ads = 5
    output = []
    for i in range(max_n):
        output.append(ads//2)
        ads = (ads//2)*3
    return output
def viralAdvertising(n, pattern):
    return sum(pattern[:n])
if __name__ == "__main__":
    n = int(input().strip())
    pattern = calc_pattern()
    result = viralAdvertising(n, pattern)
    print(result)
EOF
def sockMerchant(n, ar):
    socks = {}
    res = 0
    for el in ar:
        if el not in socks.keys():
            socks[el] = 1
        else:
            socks[el] += 1
    for key in socks.keys():
        res += socks[key]//2
    return res
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = sockMerchant(n, ar)
print(result)
EOF
def jump(c):
    res = 0
    ind = 0
    while ind != len(c)-1:
        if ind != len(c)-2 and c[ind+2] == 0:
            ind += 2
        else:
            ind += 1
        res += 1
    return res
if __name__ == "__main__":
    n = int(input().strip())
    c = list(map(int, input().strip().split(' ')))
    result = jump(c)
    print(result)
EOF
def get_grid(number):
    root = sqrt(number)
    x = int(root//1)
    y = ceil(root)
    while x*y < number:
        if x <= y:
            x += 1
        else:
            y += 1
    return (x, y)
def encryption(string):
    string = string.strip().replace(' ', '')
    str_len = len(string)
    x, y = get_grid(str_len)
    grid = [ [ '' for i in range(x) ] for _j in range(y) ]
    count = 0
    x_ind = 0
    y_ind = 0
    for ind in range(str_len):
        if count / y == 1 and count % y == 0:
            count = 0
            y_ind += 1
            x_ind = 0
        grid[x_ind][y_ind] = string[ind]
        count += 1
        x_ind += 1
    out = ''
    for _i in range(y):
        for _j in range(x):
            out += grid[_i][_j]
        out += ' '
    return out
if __name__ == "__main__":
    s = input().strip()
    result = encryption(s)
    print(result)
EOF
def substrCount(n, s):
    squashed = []
    cur = s[0]
    cnt = 0
    res = 0
    for el in s:
        if el != cur:
            squashed.append((cur, cnt))
            cur = el
            cnt = 1
        else:
            cnt += 1
    squashed.append((cur, cnt))
    print(squashed)
    for el in squashed:
        res += (el[1]*(1 + el[1]))//2
    for ind in range(len(squashed)-2):
        if squashed[ind][0] == squashed[ind+2][0] and squashed[ind+1][1] == 1:
            res += min(squashed[ind][1], squashed[ind+2][1])
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    s = input()
    result = substrCount(n, s)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
sys.setrecursionlimit(20000)
def mature_check(passwords, attempt):
    allpass = "".join(passwords)
    res = True
    for letter in attempt:
        if letter not in allpass:
            res = False
            break
    return res
def password_cracker(passwords, attempt, solution, memo):
    if len(attempt) == 0:
        return True
    if attempt in memo:
        return False
    for psw in passwords:
        if attempt.startswith(psw):
            solution.append(psw)
            memo[attempt] = True
            if password_cracker(passwords, attempt[len(psw):], solution, memo) == True:
                return True
            solution.pop()
    return False
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = int(input().strip())
        passwords = input().strip().split(' ')
        attempt = input().strip()
        solution = []
        memo = {}
        if mature_check(passwords, attempt) and password_cracker(passwords, attempt, solution, memo):
            print(" ".join(solution))
        else:
            print("WRONG PASSWORD")
EOF
if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    array = [0] * (n+1)
    for a0 in range(m):
        a, b, k = input().strip().split(' ')
        a, b, k = [int(a), int(b), int(k)]
        array[a-1] += k
        if b+1 <= n:
            array[b] -= k
    res_max = 0
    res = 0
    for dif in array:
        res += dif
        res_max = max(res_max, res)
    print(res_max)
EOF
def is_valid(a, b, c):
    if a < b+c and b < c+a and c < a+b:
        return True
    else:
        return False
def maximumPerimeterTriangle(sticks):
    res = [-1]
    sticks = sorted(sticks, reverse=True)
    print(sticks)
    for ind in range(2, len(sticks)):
        for jnd in range(1, ind):
            for knd in range(0, jnd):
                print("checking {} {} {}".format(sticks[ind], sticks[jnd], sticks[knd]))
                if is_valid(sticks[ind], sticks[jnd], sticks[knd]):
                    print("valid")
                    res = (sticks[ind], sticks[jnd], sticks[knd])
                    return res
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    sticks = list(map(int, input().rstrip().split()))
    result = maximumPerimeterTriangle(sticks)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
EOF
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def preorderTraversal(self, root):
        result, curr = [], root
        while curr:
            if curr.left is None:
                result.append(curr.val)
                curr = curr.right
            else:
                node = curr.left
                while node.right and node.right != curr:
                    node = node.right
                if node.right is None:
                    result.append(curr.val)
                    node.right = curr
                    curr = curr.left
                else:
                    node.right = None
                    curr = curr.right
        return result
class Solution2(object):
    def preorderTraversal(self, root):
        result, stack = [], [(root, False)]
        while stack:
            root, is_visited = stack.pop()
            if root is None:
                continue
            if is_visited:
                result.append(root.val)
            else:
                stack.append((root.right, False))
                stack.append((root.left, False))
                stack.append((root, True))
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def diameterOfBinaryTree(self, root):
        self.result = 0
        def helper(node):       
            if not node:
                return -1       
            left = helper(node.left)
            right = helper(node.right)
            self.result = max(self.result, 2 + left + right)
            return max(1 + left, 1 + right)
        helper(root)        
        return self.result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def calculate(self, s):
        stack = []
        num = 0
        op = '+'
        for i, c in enumerate(s):
            if c.isdigit():                 
                num = num*10 + int(c)
            if (not c.isdigit() and c != ' ') or i == len(s)-1:    
                if op == '+':               
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    stack.append(stack.pop() * num)
                else:   
                    left = stack.pop()
                    stack.append(left // num)
                    if left // num < 0 and left % num != 0:
                        stack[-1] += 1      
                num = 0     
                op = c
        return sum(stack)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def twoSum(self, numbers, target):
        left, right = 0, len(numbers)-1
        while True:
            pair_sum = numbers[left] + numbers[right]
            if pair_sum == target:
                return [left+1, right+1]
            if pair_sum < target:
                left += 1
            else:
                right -= 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def mincostToHireWorkers(self, quality, wage, K):
        wage_per_quality = [(w / float(q), q) for w, q in zip(wage, quality)]
        wage_per_quality.sort()
        workers = [-q for _, q in wage_per_quality[:K]]         
        heapq.heapify(workers)
        total_quality = -sum(workers)
        cost = wage_per_quality[K - 1][0] * total_quality       
        for wpq, q in wage_per_quality[K:]:
            heapq.heappush(workers, -q)
            total_quality += q + heapq.heappop(workers)         
            cost = min(cost, wpq * total_quality)
        return cost
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def isLongPressedName(self, name, typed):
        typed_i, name_i = 0, 0              
        while name_i < len(name):
            c, c_count = name[name_i], 1    
            name_i += 1
            while name_i < len(name) and name[name_i] == c:         
                name_i += 1
                c_count += 1
            while typed_i < len(typed) and typed[typed_i] == c:     
                typed_i += 1
                c_count -= 1
            if c_count > 0:                 
                return False
        return typed_i == len(typed)        
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestCompletingWord(self, licensePlate, words):
        letters = [c.lower() for c in licensePlate if c > "9"]  
        freq = Counter(letters)
        words.sort(key=lambda x: len(x))
        for word in words:
            if len(word) < len(letters):                        
                continue
            word_freq = Counter(word)
            for c, count in freq.items():
                if c not in word_freq or word_freq[c] < count:
                    break
            else:
                return word
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def originalDigits(self, s):
        digit_freq = [0] * 10
        letter_freq = Counter(s)
        words = [("z", [], 0), ("w", [], 2), ("u", [], 4), ("x", [], 6), ("g", [], 8),
                 ("o", [0, 2, 4], 1), ("r", [0, 4], 3), ("f", [4], 5), ("v", [5], 7), ("i", [5, 6, 8], 9)]
        for letter, other_digits, digit in words:
            word_count = letter_freq[letter]
            for other_digit in other_digits:
                word_count -= digit_freq[other_digit]
            digit_freq[digit] = word_count
        result = []
        for digit, count in enumerate(digit_freq):
            result += [str(digit)] * count
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def mostCommonWord(self, paragraph, banned):
        banned = set(banned)
        punct = {"!", "?", ",", ".", ";", "'"}
        counter = defaultdict(int)
        for word in (s.lower() for s in paragraph.split(" ")):
            word = "".join(c for c in word if c not in punct)
            if word not in banned:
                counter[word] += 1
        return max(counter.items(), key=lambda x: x[1])[0]  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def distributeCoins(self, root):
        def helper(node, parent):
            if not node:
                return 0
            left = helper(node.left, node)          
            right = helper(node.right, node)
            upshift = node.val - 1                  
            if upshift != 0:                        
                parent.val += upshift
            node.val = 1                            
            return left + right + abs(upshift)      
        return helper(root, None)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numSimilarGroups(self, A):
        N, W = len(A), len(A[0])
        word_swap = defaultdict(set)
        if N < 2 * W:           
            for i, w1 in enumerate(A):
                for j in range(i + 1, len(A)):
                    w2 = A[j]
                    if len([True for c1, c2 in zip(w1, w2) if ord(c1) - ord(c2) != 0]) == 2:
                        word_swap[w1].add(w2)
                        word_swap[w2].add(w1)
        else:
            A_set = set(A)
        def get_neighbours(a):
            if word_swap:
                return word_swap[a]
            neighbours = set()
            for i in range(W - 1):
                for j in range(i + 1, W):
                    if a[i] != a[j]:
                        neighbour = a[:i] + a[j] + a[i + 1:j] + a[i] + a[j + 1:]
                        if neighbour in A_set:
                            neighbours.add(neighbour)
            return neighbours
        groups = 0
        visited = set()
        def dfs(w):
            visited.add(w)
            for nbor in get_neighbours(w):
                if nbor not in visited:
                    dfs(nbor)
        for word in A:
            if word in visited:
                continue
            groups += 1
            dfs(word)
        return groups
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkPossibility(self, nums):
        modified = False
        for i, num in enumerate(nums[1:], 1):
            if num < nums[i - 1]:
                if modified:
                    return False
                if i != 1 and nums[i - 2] > nums[i]:    
                    nums[i] = nums[i - 1]
                modified = True
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isSubtree(self, s, t):
        def serialize(node):
            if not node:
                serial.append("
                return
            serial.append(",")
            serial.append(str(node.val))
            serialize(node.left)
            serialize(node.right)
        serial = []         
        serialize(s)
        s_serial = "".join(serial)
        serial = []
        serialize(t)
        t_serial = "".join(serial)
        return t_serial in s_serial
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def averageOfLevels(self, root):
        nodes = [root]
        result = []
        while True:
            row_sum, row_count = 0, 0       
            new_nodes = []
            for node in nodes:
                if not node:                
                    continue
                row_sum += node.val
                row_count += 1
                new_nodes.append(node.left) 
                new_nodes.append(node.right)
            if row_count == 0:
                break
            result.append(row_sum / float(row_count))
            nodes = new_nodes               
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minEatingSpeed(self, piles, H):
        bananas, max_pile = sum(piles), max(piles)
        min_rate = (bananas + H - 1) // H           
        max_rate = max_pile
        while min_rate < max_rate:
            rate = (min_rate + max_rate) // 2
            time = 0
            for pile in piles:
                time += (pile + rate - 1) // rate   
                if time > H:                        
                    break
            if time > H:                            
                min_rate = rate + 1
            else:                                   
                max_rate = rate
        return min_rate
EOF
english_total, english_roll_no = int(input()), set(map(int, input().split()))
french_total, french_roll_no = int(input()), set(map(int, input().split()))
print(len(english_roll_no.intersection(french_roll_no)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reverseVowels(self, s):
        vowels = {"a", "e", "i", "o", "u"}
        vowels |= {c.upper() for c in vowels}
        vowel_i = [i for i, c in enumerate(s) if c in vowels]
        n_vowel = len(vowel_i)
        s = [c for c in s]
        for j in range(n_vowel // 2):
            s[vowel_i[j]], s[vowel_i[n_vowel - j - 1]] = s[vowel_i[n_vowel - j - 1]], s[vowel_i[j]]
        return "".join(s)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canPartition(self, nums):
        sum_nums = sum(nums)
        if sum_nums % 2 == 1:
            return False
        nums.sort(reverse = True)               
        target = sum_nums // 2
        subset_sum = [True] + [False] * target
        for num in nums:
            for i in range(target - 1, -1, -1): 
                if subset_sum[i] and i + num <= target:
                    if i + num == target:       
                        return True
                    subset_sum[i + num] = True  
        return False
class Solution2(object):
    def canPartition(self, nums):
        nums_sum = sum(nums)
        if nums_sum % 2 == 1:
            return False
        freq = Counter(nums)
        return self.partition(freq, nums_sum // 2)
    def partition(self, freq, target):
        if target == 0:
            return True
        if target < 0:
            return False
        for num in freq:
            if freq[num] == 0:
                continue
            freq[num] -= 1      
            if self.partition(freq, target - num):
                return True
            freq[num] += 1      
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def convertToBase7(self, num):
        negative = num < 0                      
        num = abs(num)
        base_7 = []
        while num:
            num, digit = divmod(num, 7)
            base_7.append(str(digit))
        if negative:
            base_7.append("-")
        return "".join(base_7[::-1]) or "0"     
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class RandomizedCollection(object):
    def __init__(self):
        self.nums = []
        self.indices = defaultdict(set)  
    def insert(self, val):
        result = True
        if val in self.indices:
            result = False
        self.nums.append(val)
        self.indices[val].add(len(self.nums) - 1)
        return result
    def remove(self, val):
        if val not in self.indices:
            return False
        i = self.indices[val].pop()     
        if not self.indices[val]:       
            del self.indices[val]
        if i == len(self.nums) - 1:     
            self.nums.pop()
        else:
            replacement = self.nums[-1]
            self.nums[i] = replacement
            self.nums.pop()
            self.indices[replacement].discard(len(self.nums))   
            self.indices[replacement].add(i)
        return True
    def getRandom(self):
        return self.nums[random.randint(0, len(self.nums) - 1)]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProfit(self, prices):
        return sum([max(prices[i]-prices[i-1], 0) for i in range(1,len(prices))])
EOF
n = int(input().strip())
values = [int(i) for i in input().split()][:n]
positive = negative = zero = 0
for i in range(n):
    if values[i] > 0: positive += 1
    elif values[i] == 0: zero += 1
    else: negative += 1
print("%.6f\n%.6f\n%.6f" % (positive/n, negative/n, zero/n))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        count = 0
        subsequences = []   
        for i in range(len(A)):
            subsequences.append(defaultdict(int))
            for j in range(i):                                  
                diff = A[i] - A[j]
                diff_count = subsequences[j].get(diff, 0)       
                count += diff_count             
                subsequences[-1][diff] += diff_count + 1        
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Codec:
    def encode(self, root):
        if not root:
            return None
        binary = TreeNode(root.val)                 
        if not root.children:
            return binary
        binary.left = self.encode(root.children[0]) 
        node = binary.left                          
        for child in root.children[1:]:             
            node.right = self.encode(child)
            node = node.right
        return binary
    def decode(self, data):
        if not data:
            return None
        nary = Node(data.val, [])                   
        node = data.left                            
        while node:                                 
            nary.children.append(self.decode(node)) 
            node = node.right                       
        return nary
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isPalindrome(self, s):
        allowed = set(string.ascii_lowercase + string.digits)
        s = [c for c in s.lower() if c in allowed]
        i, j = 0, len(s)-1
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def judgeSquareSum(self, c):
        a = 0
        while a <= sqrt(c / 2):
            b = sqrt(c - a ** 2)
            if int(b) == b:
                return True
            a += 1
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countAndSay(self, n):
        sequence = [1]
        for _ in range(n-1):
            next = []
            for num in sequence:
                if not next or next[-1] != num:
                    next += [1, num]
                else:
                    next[-2] += 1
            sequence = next
        return "".join(map(str, sequence))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def firstMissingPositive(self, nums):
        i = 0
        while i < len(nums):
            while nums[i] > 0 and nums[i] <= len(nums) and nums[nums[i]-1] != nums[i]:
                temp = nums[nums[i]-1]
                nums[nums[i]-1] = nums[i]
                nums[i] = temp
            i += 1
        for i, num in enumerate(nums):
            if nums[i] != i+1:
                return i+1
        return len(nums)+1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLength(self, A, B):
        def mutual_subarray(length):
            subarrays = set(tuple(A[i:i + length])
                            for i in range(len(A) - length + 1))
            return any(tuple(B[j:j + length]) in subarrays
                       for j in range(len(B) - length + 1))
        low, high = 0, min(len(A), len(B)) + 1
        while low < high:   
            mid = (low + high) // 2
            if mutual_subarray(mid):    
                low = mid + 1
            else:
                high = mid              
        return low - 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def pourWater(self, heights, V, K):
        heights = [float("inf")] + heights + [float("inf")]     
        K += 1
        while V > 0:                                            
            V -= 1
            i = K
            lowest, lowest_i = heights[K], K
            while heights[i - 1] <= lowest:                     
                i -= 1
                if heights[i] < lowest:                         
                    lowest, lowest_i = heights[i], i
            if lowest < heights[K]:                             
                heights[lowest_i] += 1
                continue
            i = K
            lowest, lowest_i = heights[K], K
            while heights[i + 1] <= lowest:
                i += 1
                if heights[i] < lowest:
                    lowest, lowest_i = heights[i], i
            if lowest < heights[K]:
                heights[lowest_i] += 1
            else:
                heights[K] += 1
        return heights[1:-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Folder(object):
    def __init__(self):
        self.children = {}      
class FileSystem(object):
    def __init__(self):
        self.root = Folder()    
        self.files = {}         
    def ls(self, path):
        path = path.split("/")
        if path[-1] in self.files:
            return [path[-1]]       
        folder = self.root
        if path[-1] != "":          
            for folder_string in path[1:]:
                folder = folder.children[folder_string]
        return sorted(list(folder.children.keys()))         
    def mkdir(self, path):
        folder = self.root
        for folder_string in path.split("/")[1:]:           
            if folder_string not in folder.children:        
                folder.children[folder_string] = Folder()
            folder = folder.children[folder_string]
    def addContentToFile(self, filePath, content):
        path = filePath.split("/")
        file_name = path[-1]
        if file_name in self.files:
            self.files[file_name] += content
        else:
            self.files[file_name] = content
            folder = self.root
            for folder_string in path[1:-1]:
                folder = folder.children[folder_string]
            folder.children[file_name] = None
    def readContentFromFile(self, filePath):
        file_name = filePath.split("/")[-1]
        return self.files[file_name]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def imageSmoother(self, M):
        rows, cols = len(M), len(M[0])
        smoothed = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                nbors, total = 0, 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if r + dr < 0 or r + dr >= rows or c + dc < 0 or c + dc >= cols:
                            continue
                        total += M[r + dr][c + dc]
                        nbors += 1
                smoothed[r][c] = total // nbors         
        return smoothed
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minDiffInBST(self, root):
        self.min_diff = float("inf")
        self.prev = float("-inf")
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            self.min_diff = min(self.min_diff, node.val - self.prev)
            self.prev = node.val
            inorder(node.right)
        inorder(root)
        return self.min_diff
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        AB = defaultdict(int)
        count = 0
        for a in A:
            for b in B:
                AB[a + b] += 1
        for c in C:
            for d in D:
                if -(c + d) in AB:
                    count += AB[-(c + d)]
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestWordDistance(self, words, word1, word2):
        last1, last2 = -1, -1   
        same = word1 == word2   
        distance = len(words)
        for i, word in enumerate(words):
            if word == word1:
                if same:
                    last1, last2 = last2, i     
                else:
                    last1 = i
            elif word == word2:
                last2 = i
            if last1 != -1 and last2 != -1:
                distance = min(distance, abs(last1 - last2))
        return distance
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def areSentencesSimilarTwo(self, words1, words2, pairs):
        def find(word):
            if word not in mapping:
                return None
            while mapping[word] != word:
                mapping[word] = mapping[mapping[word]]      
                word = mapping[word]                        
            return word
        if len(words1) != len(words2):                      
            return False
        mapping = {}                                        
        for w1, w2 in pairs:
            p1, p2 = find(w1), find(w2)
            if p1:
                if p2:
                    mapping[p1] = p2                        
                else:
                    mapping[w2] = p1                        
            else:
                if p2:
                    mapping[w1] = p2                        
                else:
                    mapping[w1] = mapping[w2] = w1          
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                continue
            p1, p2 = find(w1), find(w2)
            if not p1 or not p2 or p1 != p2:                
                return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMinMoves(self, machines):
        dresses = sum(machines)
        if dresses % len(machines) != 0:    
            return -1
        target = dresses // len(machines)
        moves, running = 0, 0
        machines = [m - target for m in machines]
        for machine in machines:
            running += machine
            moves = max(moves, abs(running), machine)
        return moves
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def champagneTower(self, poured, query_row, query_glass):
        glasses = [poured]
        for row in range(query_row):
            new_glasses = [0 for _ in range(len(glasses) + 1)]
            for i, glass in enumerate(glasses):
                pour = max(glass - 1, 0) / 2.0
                new_glasses[i] += pour
                new_glasses[i + 1] += pour
            glasses = new_glasses
        return min(glasses[query_glass], 1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shiftingLetters(self, S, shifts):
        s = [ord(c) - ord("a") for c in S]              
        cumulative_shift = 0
        for i in range(len(s) - 1, -1, -1):
            cumulative_shift += shifts[i]
            s[i] = (s[i] + cumulative_shift) % 26       
        return "".join(chr(c + ord("a")) for c in s)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isConvex(self, points):
        points.append(points[0])  
        points.append(points[1])  
        previous = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
        previous_cross = 0
        for i in range(2, len(points)):
            vector = [points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]]
            cross_product = vector[0] * previous[1] - vector[1] * previous[0]
            if cross_product != 0:
                if previous_cross * cross_product < 0:
                    return False
                previous_cross = cross_product
            previous = vector
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numIslands2(self, m, n, positions):
        island_count = [0]
        parent = {}         
        for r, c in positions:
            if (r, c) in parent:    
                island_count.append(island_count[-1])
                continue
            nbors = set()
            for nbor in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
                if nbor in parent:
                    island = parent[nbor]
                    while island != parent[island]:
                        parent[island] = parent[parent[island]]     
                        island = parent[island]
                    nbors.add(island)
            if not nbors:
                parent[(r, c)] = (r, c)             
                island_count.append(island_count[-1] + 1)
            else:
                this_island = nbors.pop()
                for nbor in nbors:
                    parent[nbor] = this_island
                parent[(r, c)] = this_island
                island_count.append(island_count[-1] - len(nbors))
        return island_count[1:]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def combinationSum2(self, candidates, target):
        results = []
        freq = list(Counter(candidates).items())
        self.combos(freq, 0, target, [], results)
        return results
    def combos(self, freq, next, target, partial, results):
        if target == 0:
            results.append(partial)
            return
        if next == len(freq):
            return
        for i in range(freq[next][1]+1):
            if i * freq[next][0] > target:
                break
            self.combos(freq, next+1, target-i*freq[next][0], partial + [freq[next][0]]*i, results)
class Solution_Iterative(object):
    def combinationSum2(self, candidates, target):
        results = []
        partials = [[]]
        freq = list(Counter(candidates).items())
        for candidate, count in freq:
            new_partials = []
            for partial in partials:
                partial_sum = sum(partial)
                for i in range(count + 1):
                    if partial_sum + candidate*i < target:
                        new_partials.append(partial + [candidate]*i)
                    elif partial_sum + candidate*i == target:
                        results.append(partial + [candidate]*i)
                    if partial_sum + candidate*i >= target:
                        break
            partials = new_partials
        return results
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Codec:
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  
    def __init__(self):
        self.map = {}       
    def encode(self, longUrl):
        encoding = []
        for i in range(6):
            encoding.append(self.letters[random.randint(0, 61)])
        encoding = "".join(encoding)
        if encoding in self.map:    
            encoding = self.encode(longUrl)
        self.map[encoding] = longUrl
        return encoding
    def decode(self, shortUrl):
        return self.map[shortUrl]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def singleNumber(self, nums):
        xor = 0
        for num in nums:
            xor = xor ^ num
        bit = 0
        while not (1 << bit) & xor:
            bit += 1
        bit_set_xor, bit_not_set_xor = 0, 0
        for num in nums:
            if (1 << bit) & num:
                bit_set_xor = bit_set_xor ^ num
            else:
                bit_not_set_xor = bit_not_set_xor ^ num
        return [bit_set_xor, bit_not_set_xor]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def tallestBillboard(self, rods):
        diffs = {0 : 0}     
        for rod in rods:
            new_diffs = defaultdict(int, diffs)
            for diff, used_len in diffs.items():
                new_diffs[diff + rod] = max(used_len + rod, new_diffs[diff + rod])
                new_diffs[abs(diff - rod)] = max(used_len + rod, new_diffs[abs(diff - rod)])
            diffs = new_diffs
        return diffs[0] // 2
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        super_ugly = [1]
        indices = [0 for _ in range(len(primes))]
        candidates = primes[:]
        while len(super_ugly) < n:
            ugly = min(candidates)
            super_ugly.append(ugly)
            for i in range(len(candidates)):    
                if ugly == candidates[i]:
                    indices[i] += 1
                    candidates[i] = primes[i] * super_ugly[indices[i]]
        return super_ugly[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findErrorNums(self, nums):
        for num in nums:
            if nums[abs(num) - 1] < 0:
                duplicate = abs(num)
            else:
                nums[abs(num) - 1] *= -1
        for i, num in enumerate(nums):
            if num > 0:
                missing = i + 1
                break
        return [duplicate, missing]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def bulbSwitch(self, n):
        return int(n**0.5)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestSubarray(self, A, K):
        n = len(A)
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + A[i]
        queue = deque()
        result = n + 1
        for i in range(n + 1):
            while queue and prefix_sums[i] - prefix_sums[queue[0]] >= K:
                result = min(result, i - queue.popleft())   
            while queue and prefix_sums[queue[-1]] >= prefix_sums[i]:
                queue.pop()                         
            queue.append(i)                         
        return result if result <= n else -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        if endWord not in wordList:
            return []
        wordList = set(wordList)                    
        left, right = {beginWord}, {endWord}        
        left_parents, right_parents = defaultdict(set), defaultdict(set)    
        swapped = False                             
        while left and right and not (left & right):    
            if len(right) < len(left):              
                left, right = right, left
                left_parents, right_parents = right_parents, left_parents
                swapped = not swapped
            next_left = defaultdict(set)
            for word in left:
                for char in string.ascii_lowercase:
                    for i in range(len(beginWord)):
                        n = word[:i] + char + word[i + 1:]  
                        if n in wordList and n not in left_parents: 
                            next_left[n].add(word)
            left_parents.update(next_left)
            left = set(next_left.keys())
        if swapped:                                 
            left, right = right, left
            left_parents, right_parents = right_parents, left_parents
        ladders = [[word] for word in left & right] 
        while ladders and ladders[0][0] not in beginWord:   
            ladders = [[p] + l for l in ladders for p in left_parents[l[0]]]
        while ladders and ladders[0][-1] != endWord:        
            ladders = [l + [p] for l in ladders for p in right_parents[l[-1]]]
        return ladders
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isAdditiveNumber(self, num):
        n = len(num)
        if n < 3:
            return False
        for second in range(1, 1 + (n - 1) // 2):       
            if num[0] == "0" and second > 1:
                break
            third = second + 1                          
            while n - third >= max(second, third - second):
                if num[second] == "0" and third > second + 1:
                    break
                n1, n2 = int(num[0:second]), int(num[second:third])
                start = third
                while True:
                    next_int = n1 + n2
                    next_start = start + len(str(next_int))
                    if num[start] == "0" and next_start > start + 1:
                        break
                    if next_int != int(num[start:next_start]):
                        break
                    if next_start == n:
                        return True
                    n1, n2, start = n2, next_int, next_start
                third += 1
        return False
EOF
d = OrderedDict()
for i in range(int(input())):
    item,price = input().rsplit(None,1)
    d[item] = d.get(item,0) + int(price)
for k, v in d.items():
    print(k,v)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isNStraightHand(self, hand, W):
        if len(hand) % W != 0:      
            return False
        if W == 1:
            return True
        hand.sort()
        partials = []               
        for num in hand:
            if not partials or partials[0][0] == num:   
                heapq.heappush(partials, (num, 1))
                continue
            if num > partials[0][0] + 1:                
                return False
            end, length = heapq.heappop(partials)       
            if length != W - 1:
                heapq.heappush(partials, (num, length + 1))
        return not partials
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def repeatedStringMatch(self, A, B):
        if set(B) - set(A):         
            return -1
        div, mod = divmod(len(B), len(A))
        if mod != 0:
            div += 1
        for i in range(2):
            if B in A * (div + i):
                return div + i
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def flipgame(self, fronts, backs):
        duplicates = {f for f, b in zip(fronts, backs) if f == b}
        result = float("inf")
        for f, b in zip(fronts, backs):
            if f != b:
                if f not in duplicates:
                    result = min(result, f)
                if b not in duplicates:
                    result = min(result, b)
        return 0 if result == float("inf") else result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestDistance(self, words, word1, word2):
        shortest = len(words)
        i_1, i_2 = float("-inf"), float("-inf")         
        for i, word in enumerate(words):
            if word == word1:
                i_1 = i
                shortest = min(shortest, i_1 - i_2)
            if word == word2:
                i_2 = i
                shortest = min(shortest, i_2 - i_1)
        return shortest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def splitIntoFibonacci(self, S):
        MAX_NUM = 2 ** 31 - 1
        def helper(i, n1, n2):          
            fib = [n1, n2]
            while i < len(S):
                next_num = fib[-1] + fib[-2]
                if next_num > MAX_NUM:
                    return []
                next_str = str(next_num)
                if S[i:i + len(next_str)] != next_str:  
                    return []
                fib.append(next_num)
                i += len(next_str)
            return fib
        for len1 in range(1, (len(S) + 1) // 2):
            if len1 > 1 and S[0] == "0":    
                return []
            n1 = int(S[:len1])
            if n1 > MAX_NUM:
                return []
            len2 = 1
            while len(S) - len1 - len2 >= max(len1, len2):
                if len2 > 1 and S[len1] == "0": 
                    break
                n2 = int(S[len1:len1 + len2])
                if n2 > MAX_NUM:
                    break
                fibonacci = helper(len1 + len2, n1, n2)
                if fibonacci:
                    return fibonacci
                len2 += 1
        return []
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def allPossibleFBT(self, N):
        memo = {}
        def helper(n):
            if n % 2 == 0:
                return []
            if n == 1:
                return [TreeNode(0)]
            if n in memo:
                return memo[n]
            result = []
            for left_size in range(1, n, 2):
                right_size = n - 1 - left_size
                left_subtrees = helper(left_size)
                right_subtrees = helper(right_size)
                for left_subtree in left_subtrees:
                    for right_subtree in right_subtrees:
                        root = TreeNode(0)
                        root.left = left_subtree
                        root.right = right_subtree
                        result.append(root)
            memo[n] = result
            return result
        return helper(N)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def solve(self, board):
        if not board or not board[0]:
            return
        rows, cols = len(board), len(board[0])
        to_expand = []              
        for row in range(rows):     
            to_expand += [(row, 0), (row, cols - 1)]
        for col in range(1, cols-1):
            to_expand += [(0, col), (rows - 1, col)]
        while to_expand:            
            row, col = to_expand.pop()
            if 0 <= row < rows and 0 <= col < cols and board[row][col] == 'O':
                board[row][col] = 'T'
                for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    to_expand.append((row + dr, col + dc))
        for row in range(rows):     
            for col in range(cols):
                if board[row][col] == 'O':
                    board[row][col] = 'X'
                elif board[row][col] == 'T':
                    board[row][col] = 'O'
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def arrayNesting(self, nums):
        visited = set()             
        longest = 0                 
        for i, num in enumerate(nums):
            if num in visited:
                continue
            current = set()         
            while num not in current:
                current.add(num)    
                num = nums[num]     
            longest = max(longest, len(current))
            if longest >= len(nums) - i - 1:    
                break
            visited |= current      
        return longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reverseWords(self, s):
        words = s.split()
        return " ".join(words[::-1])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def buildTree(self, preorder, inorder):
        def build(stop):
            if not inorder or inorder[-1] == stop:
                return None
            root_val = preorder.pop()
            root = TreeNode(root_val)
            root.left = build(root_val)     
            inorder.pop()                   
            root.right = build(stop)        
            return root
        preorder.reverse()      
        inorder.reverse()
        return build(None)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minMalwareSpread(self, graph, initial):
        best_reduction = 0                  
        best_node = min(initial)            
        initial = set(initial)              
        def connected(node):                
            if node in group:
                return
            group.add(node)
            [connected(nbor) for nbor, linked in enumerate(graph[node]) if linked == 1] 
        visited = set()                     
        for node in range(len(graph)):
            if node in visited:
                continue
            group = set()
            connected(node)
            overlap = initial & group       
            if len(overlap) == 1 and len(group) > best_reduction:
                best_reduction = len(group)
                best_node = overlap.pop()
            visited |= group
        return best_node
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def subsetsWithDup(self, nums):
        num_count = Counter(nums)
        results = [[]]
        for num in num_count:
            results += [partial+[num]*i for i in range(1,num_count[num]+1) for partial in results]
        return results
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isValid(self, s):
        stack = []
        match = {'(' : ')', '[' : ']', '{' : '}'}
        for c in s:
            if c in match:
                stack.append(c)
            else:
                if not stack or match[stack.pop()] != c:
                    return False
        return not stack
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
class DLL:
    def __init__(self):
        self.head = Node(None, None)      
        self.tail = Node(None, None)      
        self.head.next = self.tail
        self.tail.prev = self.head
    def insert(self, node):
        node.prev, self.tail.prev.next = self.tail.prev, node
        node.next, self.tail.prev = self.tail, node
    def remove_at_head(self):
        node = self.head.next
        node.next.prev = self.head
        self.head.next = self.head.next.next
        key = node.key
        del node
        return key
    def update(self, node):
        node.prev.next = node.next      
        node.next.prev = node.prev
        self.insert(node)               
class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = DLL()
        self.mapping = {}
    def get(self, key):
        if key not in self.mapping:
            return -1
        node = self.mapping[key]
        self.queue.update(node)
        return node.val
    def set(self, key, value):
        if key in self.mapping:         
            node = self.mapping[key]
            node.val = value
            self.queue.update(node)
            return
        node = Node(key, value)         
        self.mapping[key] = node
        self.queue.insert(node)
        if self.capacity == 0:          
            removed_key = self.queue.remove_at_head()
            del self.mapping[removed_key]
        else:                           
            self.capacity -= 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def circularArrayLoop(self, nums):
        n = len(nums)
        for i, num in enumerate(nums):
            pos = num > 0               
            j = (i + num) % n           
            steps = 1
            while steps < n and nums[j] % n != 0 and (nums[j] > 0) == pos:
                j = (j + nums[j]) % n   
                steps += 1
            if steps == n:              
                return True
            nums[i] = 0
            j = (i + num) % n           
            while nums[j] % n != 0 and (nums[j] > 0) == pos:
                j, nums[j] = (j + nums[j]) % n, 0
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isPalindrome(self, head):
        fast, slow = head, head
        rev = None                      
        while fast and fast.next:
            fast = fast.next.next
            next_slow = slow.next
            slow.next = rev
            rev = slow
            slow = next_slow
        if fast:
            slow = slow.next
        while slow:
            if slow.val != rev.val:
                return False
            slow = slow.next
            rev = rev.next
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def leafSimilar(self, root1, root2):
        def inorder(node):
            if node.left:                           
                yield from inorder(node.left)
            if not node.left and not node.right:    
                yield node.val
            if node.right:                          
                yield from inorder(node.right)
        leaves = itertools.zip_longest(inorder(root1), inorder(root2))
        return all(l1 == l2 for l1, l2 in leaves)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numMatchingSubseq(self, S, words):
        letter_to_suffixes = defaultdict(list)  
        letter_to_suffixes["
        result = 0
        for c in "
            suffixes = letter_to_suffixes[c]    
            del letter_to_suffixes[c]
            for suffix in suffixes:
                if len(suffix) == 0:            
                    result += 1
                    continue
                next_letter, next_suffix = suffix[0], suffix[1:]
                letter_to_suffixes[next_letter].append(next_suffix)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def widthOfBinaryTree(self, root):
        if not root:
            return 0
        max_width = 1
        nodes = [(root, 0)]     
        while True:
            new_nodes = []
            for node, i in nodes:
                if node.left:
                    new_nodes.append((node.left, i * 2))
                if node.right:
                    new_nodes.append((node.right, i * 2 + 1))
            if not new_nodes:
                break
            nodes = new_nodes
            max_width = max(max_width, 1 + nodes[-1][1] - nodes[0][1])
        return max_width
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def plusOne(self, head):
        new_head = ListNode(0)      
        new_head.next = head
        i, j = new_head, new_head   
        while i.next:               
            i = i.next
            if i.val != 9:          
                j = i
        j.val += 1                  
        j = j.next
        while j:                    
            j.val = 0
            j = j.next
        if new_head.val == 0:       
            return head
        return new_head
EOF
class Solution(object):
    def canPermutePalindrome(self, s):
        return sum(v % 2 for v in collections.Counter(s).values()) < 2
EOF
symbols = string.ascii_lowercase
def gen_uniforms(s):
    uniforms = {}
    mult = 1
    let_prev = ''
    for let in s:
        if let == let_prev:
            mult += 1
        else:
            mult = 1
        let_prev = let
        uniforms[((symbols.index(let) + 1)*mult)] = True
    return uniforms
if __name__ == "__main__":
    s = input().strip()
    n = int(input().strip())
    uniforms = gen_uniforms(s)
    for a0 in range(n):
        x = int(input().strip())
        if x in uniforms:
            print("Yes")
        else:
            print("No")
EOF
numbers = "0123456789"
lower_case = "abcdefghijklmnopqrstuvwxyz"
upper_case = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
special_characters = "!
def minimumNumber(n, password):
    res = 0
    if not any(x in numbers for x in password):
        res += 1
    if not any(x in lower_case for x in password):
        res += 1
    if not any(x in upper_case for x in password):
        res += 1
    if not any(x in special_characters for x in password):
        res += 1
    if len(password) < 6:
        return max(res, 6 - len(password))
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    password = input()
    answer = minimumNumber(n, password)
    fptr.write(str(answer) + '\n')
    fptr.close()
EOF
def pickingNumbers(arr):
    arr_s = sorted(arr)
    res = 1
    cur = 1
    diff = 0
    for ind in range(1, len(arr_s)):
        diff += arr_s[ind] - arr_s[ind - 1]
        if diff > 1:
            res = max(res, cur)
            cur = 1
            diff = 0
        else:
            cur += 1
    res = max(res, cur)
    return res
if __name__ == "__main__":
    n = int(input().strip())
    a = list(map(int, input().strip().split(' ')))
    result = pickingNumbers(a)
    print(result)
EOF
def solve(arr, money):
    cost_map = {}
    for i, cost in enumerate(arr):
        johnny = money - cost
        if johnny in cost_map.keys():
            print("{} {}".format(cost_map[johnny]+1, i+1))
        else:
            cost_map[cost] = i
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        money = int(input().strip())
        n = int(input().strip())
        arr = list(map(int, input().strip().split(' ')))
        solve(arr, money)
EOF
def lca(root , v1 , v2):
    vals = sorted([v1, v2])
    v1, v2 = vals[0], vals[1]
    node = root
    while True:
        if v1 < v2 < node.data:
            node = node.left
        if v1 > v2 > node.data:
            node = node.right
        if v1 <= node.data <= v2:
            break
    return node
EOF
def alternatingCharacters(s):
    string = list(s)
    last = string.pop()
    res = 0
    while string:
        newone = string.pop()
        if newone == last:
            res += 1
        else:
            last = newone
    return res
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = alternatingCharacters(s)
    print(result)
EOF
dict_mag = dict()
dict_ran = dict()
def ransom_note(magazine, ransom):
    for word in magazine:
        if word in dict_mag.keys():
            dict_mag[word] += 1
        else:
            dict_mag[word] = 1
    for word in ransom:
        if word in dict_ran.keys():
            dict_ran[word] += 1
        else:
            dict_ran[word] = 1
    for key in dict_ran.keys():
        if dict_ran[key] > dict_mag[key]:
            return False
    return True
m, n = map(int, input().strip().split(' '))
magazine = input().strip().split(' ')
ransom = input().strip().split(' ')
answer = ransom_note(magazine, ransom)
if(answer):
    print("Yes")
else:
    print("No")
EOF
def diagonalDifference(a):
    n = len(a)
    diag_pr = diag_sec = 0
    for i in range(n):
        diag_pr += a[i][i]
        diag_sec += a[n-i-1][i]
    return abs(diag_pr - diag_sec)
if __name__ == "__main__":
    n = int(input().strip())
    a = []
    for a_i in range(n):
       a_t = [int(a_temp) for a_temp in input().strip().split(' ')]
       a.append(a_t)
    result = diagonalDifference(a)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def splitArray(self, nums, m):
        left, right = max(nums), sum(nums)  
        while left < right:  
            mid = (left + right) // 2
            if self.can_split(nums, m, mid):  
                right = mid     
            else:
                left = mid + 1
        return left
    def can_split(self, nums, m, max_subarray):  
        subarray_sum = 0
        for num in nums:
            if num + subarray_sum > max_subarray:  
                m -= 1  
                if m <= 0:  
                    return False
                subarray_sum = num  
            else:
                subarray_sum += num
        return True
EOF
class Solution(object):
    def isAnagram(self, s, t):
        if len(s) != len(t):
            return False
        count = collections.defaultdict(int)
        for c in s:
            count[c] += 1
        for c in t:
            count[c] -= 1
            if count[c] < 0:
                return False
        return True
class Solution2(object):
    def isAnagram(self, s, t):
        return sorted(s) == sorted(t)
class Solution3(object):
    def isAnagram(self, s, t):
        return collections.Counter(s) == collections.Counter(t)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findRedundantConnection(self, edges):
        parents = {}
        def find_parent(n):
            if n not in parents:                        
                return n
            parents[n] = find_parent(parents[n])        
            return parents[n]                           
        for a, b in edges:
            parent_a, parent_b = find_parent(a), find_parent(b)
            if parent_a == parent_b:
                return [a, b]
            parents[parent_a] = parent_b                
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def racecar(self, target):
        min_steps = {0: 0}          
        def helper(dist):
            if dist in min_steps:
                return min_steps[dist]
            k = dist.bit_length()
            if 2 ** k - 1 == dist:                          
                return k
            steps = k + 1 + helper(2 ** k - 1 - dist)       
            for j in range(k - 1):                          
                steps = min(steps, k + j + 1 + helper(dist - 2 ** (k - 1) + 2 ** j))
            min_steps[dist] = steps
            return steps
        return helper(target)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def preimageSizeFZF(self, K):
        def factorial_zeros(n):         
            factor = 5
            result = 0
            while factor <= n:
                result += n // factor
                factor *= 5
            return result
        left, right = 0, 10 * K         
        while left < right:             
            mid = (left + right) // 2
            mid_zeros = factorial_zeros(mid)
            if mid_zeros < K:           
                left = mid + 1
            else:                       
                right = mid
        return 5 if factorial_zeros(right) == K else 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxA(self, N):
        def helper(n):
            if n in memo:
                return memo[n]
            max_A = n       
            for i in range(max(n - 5, 0), n - 3):               
                max_A = max(max_A, helper(i) * (n - i - 1))     
            memo[n] = max_A
            return max_A
        memo = {}
        return helper(N)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sequenceReconstruction(self, org, seqs):
        extended = [None] + org  
        pairs = set((n1, n2) for n1, n2 in zip(extended, org))
        num_to_index = {num: i for i, num in enumerate(extended)}
        for seq in seqs:
            for n1, n2 in zip([None] + seq, seq):
                if n2 not in num_to_index or num_to_index[n2] <= num_to_index[n1]:
                    return False
                last_index = num_to_index[n2]
                pairs.discard((n1, n2))
        return not pairs
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isBalanced(self, root):
        def balanced(node):
            if not node:
                return 0
            left_depth = balanced(node.left)
            right_depth = balanced(node.right)
            if left_depth == -1 or right_depth == -1:
                return -1
            if abs(left_depth - right_depth) > 1:
                return -1
            return 1 + max(left_depth, right_depth)
        return balanced(root) != -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def spiralMatrixIII(self, R, C, r0, c0):
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0]]      
        r, c = r0, c0
        direction = 0
        result = [[r0, c0]]
        side = 1                                        
        while len(result) < R * C:
            dr, dc = moves[direction]
            for _ in range(side):                       
                r += dr
                c += dc
                if 0 <= r < R and 0 <= c < C:           
                    result.append([r, c])
            direction = (direction + 1) % 4             
            dr, dc = moves[direction]
            for _ in range(side):
                r += dr
                c += dc
                if 0 <= r < R and 0 <= c < C:
                    result.append([r, c])
            direction = (direction + 1) % 4
            side += 1                                   
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        traversal = []
        level = [root]
        forward = True
        while level:
            new_level = []
            if forward:
                traversal.append([n.val for n in level])
            else:
                traversal.append([n.val for n in level[::-1]])
            for node in level:
                if node.left:
                    new_level.append(node.left)
                if node.right:
                    new_level.append(node.right)
            level = new_level
            forward = not forward
        return traversal
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findPaths(self, m, n, N, i, j):
        paths = 0
        dp = [[0 for _ in range(n)] for _ in range(m)]  
        dp[i][j] = 1  
        for _ in range(N):
            new_dp = [[0 for _ in range(n)] for _ in range(m)]
            for r in range(m):
                for c in range(n):
                    if r == 0:
                        paths += dp[r][c]
                    if r == m - 1:
                        paths += dp[r][c]
                    if c == 0:
                        paths += dp[r][c]
                    if c == n - 1:
                        paths += dp[r][c]
                    paths %= 10 ** 9 + 7
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        if 0 <= r + dr < m and 0 <= c + dc < n:
                            new_dp[r + dr][c + dc] += dp[r][c]
                    new_dp[r][c] %= 10 ** 9 + 7
            dp = new_dp
        return paths
class Solution2(object):
    def findPaths(self, m, n, N, i, j):
        def helper(r, c, steps):
            if steps == 0:
                return 0
            if (r, c, steps) in memo:
                return memo[(r, c, steps)]
            paths = 0
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= r + dr < m and 0 <= c + dc < n:
                    paths += helper(r + dr, c + dc, steps - 1)
                else:
                    paths += 1  
                paths %= 10 ** 9 + 7
            memo[(r, c, steps)] = paths
            return paths
        memo = {}
        return helper(i, j, N)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numSpecialEquivGroups(self, A):
        def canonical(s):
            evens = sorted([s[i] for i in range(0, len(s), 2)])
            odds = sorted([s[i] for i in range(1, len(s), 2)])
            return "".join(evens + odds)
        return len({canonical(s) for s in A})
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def smallestFromLeaf(self, root):
        def helper(node):
            if not node:            
                return ""
            node_char = chr(node.val + ord("a"))
            left, right = helper(node.left), helper(node.right)
            if not left or not right:
                return left + right + node_char     
            return left + node_char if left < right else right + node_char
        return helper(root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def strongPasswordChecker(self, s):
        upper, lower, digit = False, False, False
        subs, i = 0, 0  
        singles, doubles = 0, 0  
        while i < len(s):
            if s[i].isdigit():
                digit = True
            if s[i].isupper():
                upper = True
            if s[i].islower():
                lower = True
            if i >= 2 and s[i] == s[i - 1] == s[i - 2]:  
                seq = 2
                while i < len(s) and s[i] == s[i - 1]:  
                    seq += 1
                    i += 1
                subs += seq // 3
                if seq % 3 == 0:
                    singles += 1
                if seq % 3 == 1:
                    doubles += 1
            else:
                i += 1
        types_missing = 3 - (digit + upper + lower)
        if len(s) < 6:
            return max(types_missing, 6 - len(s))
        if len(s) <= 20:
            return max(types_missing, subs)
        deletions = len(s) - 20
        subs -= min(deletions, singles)
        subs -= min(max(deletions - singles, 0), doubles * 2) / 2
        subs -= max(deletions - singles - 2 * doubles, 0) / 3
        return deletions + max(types_missing, subs)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def searchInsert(self, nums, target):
        left = 0
        right = len(nums)
        while left <= right and left < len(nums) and right >= 0:
            mid = (left + right) // 2
            if target == nums[mid]:
                return mid
            if target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return left
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minPatches(self, nums, n):
        next_missing = 1
        patches = 0
        i = 0
        while next_missing <= n:
            if i < len(nums) and nums[i] <= next_missing:
                next_missing += nums[i]
                i += 1
            else:
                next_missing += next_missing
                patches += 1
        return patches
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def prisonAfterNDays(self, cells, N):
        day = 0
        state = tuple(cells)                
        state_to_day = {}
        def next_state(state):
            return tuple([0] + [int(not (state[i - 1] ^ state[i + 1])) for i in range(1, 7)] + [0])
        while day < N and state not in state_to_day:    
            state_to_day[state] = day
            day += 1
            state = next_state(state)
        if day < N:
            cycle = day - state_to_day[state]
            remaining = (N - state_to_day[state]) % cycle
            for _ in range(remaining):
                state = next_state(state)
        return list(state)
EOF
n = int(input().strip())
A = [int(A_temp) for A_temp in input().strip().split(' ')]
a = []
d = 0
for i in range(len(A)):
    for j in range(len(A)):
        if (A[i] == A[j] and i != j):
            d = abs(i-j)
            a.append(d)
if not a:
    print('-1')
else:
    print(min(a))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def splitBST(self, root, V):
        def splitter(node):
            if not node:
                return [None, None]
            if V < node.val:
                less, more = splitter(node.left)    
                node.left = more                    
                return [less, node]
            less, more = splitter(node.right)       
            node.right = less
            return [node, more]
        return splitter(root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def calPoints(self, ops):
        points = []
        for op in ops:
            if op == "+":               
                points.append(points[-1] + points[-2])
            elif op == "D":             
                points.append(2 * points[-1])
            elif op == "C":             
                points.pop()
            else:
                points.append(int(op))
        return sum(points)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def twoSum(self, nums, target):
        num_to_index = {}           
        for i, num in enumerate(nums):
            if target - num in num_to_index:
                return [num_to_index[target - num], i]
            num_to_index[num] = i
        return []   
EOF
a = [int(i) for i in input().split()]
b = [int(i) for i in input().split()]
alice = bob = 0
for i in range(0, 3):
    if a[i] > b[i]:
        alice += 1
    elif a[i] < b[i]:
        bob += 1
print("%s %s" % (alice, bob))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        consecutive, max_consecutive = 0, 0
        for num in nums:
            if num == 0:
                max_consecutive = max(max_consecutive, consecutive)
                consecutive = 0
            else:
                consecutive += 1
        return max(max_consecutive, consecutive)    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maskPII(self, S):
        if "            name, address = S.lower().split("            return name[0] + "*****" + name[-1] + "
        digits = [c for c in S if "0" <= c <= "9"]          
        country, local = digits[:-10], digits[-10:]         
        result = []
        if country:
            result = ["+"] + ["*"] * len(country) + ["-"]   
        result += ["***-***-"] + local[-4:]                 
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def permute(self, nums):
        permutations = [[]]
        for num in nums:
            new_permutations = []
            for perm in permutations:
                for i in range(len(perm) + 1):
                    new_permutations.append(perm[:i] + [num] + perm[i:])
            permutations = new_permutations
        return permutations
class Solution2(object):
    def permute(self, nums):
        return self.permute_helper(nums, 0)
    def permute_helper(self, nums, index):
        permutations = []
        if index >= len(nums):
            permutations.append(nums[:])        
        for i in range(index, len(nums)):
            nums[i], nums[index] = nums[index], nums[i]     
            permutations += self.permute_helper(nums, index + 1)
            nums[i], nums[index] = nums[index], nums[i]     
        return permutations
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reachingPoints(self, sx, sy, tx, ty):
        while tx > sx and ty > sy:
            tx, ty = tx % ty, ty % tx           
        if tx == sx and (ty - sy) % sx == 0:
            return True
        return ty == sy and (tx - sx) % sy == 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def boldWords(self, words, S):
        S = "
        bold = [False for _ in range(len(S))]   
        for word in words:
            i = S.find(word, 1)
            while i != -1:                      
                bold[i:i + len(word)] = [True] * len(word)      
                i = S.find(word, i + 1)         
        result = []
        for i in range(len(S)):
            if bold[i] and not bold[i - 1]:     
                result.append("<b>")
            elif not bold[i] and bold[i - 1]:   
                result.append("</b>")
            result.append(S[i])
        result = result[1:-1]                   
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        if k <= 0 or t < 0: 
            return False    
        buckets = {}        
        for i, num in enumerate(nums):
            bucket = num // (t + 1)     
            if bucket in buckets:
                return True
            if bucket+1 in buckets and abs(num - buckets[bucket+1]) <= t:   
                return True
            if bucket-1 in buckets and abs(num - buckets[bucket-1]) <= t:
                return True
            buckets[bucket] = num       
            if i - k >= 0:              
                old_bucket = nums[i - k] // (t + 1)
                del buckets[old_bucket]
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def evaluate(self, expression):
        tokens = expression.split(" ")          
        scopes = [{}]                           
        def helper(start):                      
            if start >= len(tokens):            
                return 0, start
            operator = tokens[start]
            if operator[0] == "(":              
                operator = operator[1:]
                scopes.append(dict(scopes[-1])) 
            closing_brackets = 0
            while operator[len(operator) - 1 - closing_brackets] == ")":    
                closing_brackets += 1
            if closing_brackets > 0:
                operator = operator[:-closing_brackets]
            if operator.isdigit() or operator[0] == "-" and operator[1:].isdigit():
                result = int(operator), start + 1
            elif operator == "add":
                left, next_i = helper(start + 1)
                right, next_i = helper(next_i)
                result = (left + right, next_i)
            elif operator == "mult":
                left, next_i = helper(start + 1)
                right, next_i = helper(next_i)
                result = (left * right, next_i)
            elif operator == "let":
                next_i = start + 1
                while continue_let(next_i):
                    variable = tokens[next_i]
                    expression, next_i = helper(next_i + 1)
                    scopes[-1][variable] = expression
                result =  helper(next_i)
            else:                               
                result = (scopes[-1][operator], start + 1)
            while closing_brackets > 0:         
                closing_brackets -= 1
                scopes.pop()
            return result
        def continue_let(i):                    
            return "a" <= tokens[i][0] <= "z" and tokens[i][-1] != ")"
        return helper(0)[0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def constructFromPrePost(self, pre, post):
        def helper(pre_start, pre_end, post_start, post_end):   
            if pre_start == pre_end:
                return None
            root = TreeNode(pre[pre_start])
            if post_end == post_start + 1:
                return root
            idx = pre_indices[post[post_end - 2]]       
            left_size = idx - pre_start - 1             
            root.left = helper(pre_start + 1, idx, post_start, post_start + left_size)
            root.right = helper(idx, pre_end, post_start + left_size, post_end - 1)
            return root
        pre_indices = {val: i for i, val in enumerate(pre)} 
        return helper(0, len(pre), 0, len(post))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def productExceptSelf(self, nums):
        products = [1]  
        for i in range(1, len(nums)):
            products.append(nums[i-1] * products[-1])
        right_product = 1
        for i in range(len(nums)-1, -1, -1):
            products[i] *= right_product
            right_product *= nums[i]
        return products
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def orderlyQueue(self, S, K):
        if K > 1:
            return "".join(sorted(S))
        return min(S[i:] + S[:i] for i in range(len(S)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findWords(self, words):
        keyboard = {}
        rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        for i, row in enumerate(rows):
            for c in row:
                keyboard[c] = i
        result = []
        for word in words:
            row = -1
            for c in word:
                if row == -1:                       
                    row = keyboard[c.lower()]
                elif keyboard[c.lower()] != row:    
                    break
            else:
                result.append(word)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def spellchecker(self, wordlist, queries):
        def replace_vowels(word):
            return sub('[aeiou]', '_', word)
        wordsset = set(wordlist)
        lower_words, vowel_words = {}, {}
        for word in wordlist:
            lower_words.setdefault(word.lower(), word)  
        for word in lower_words.keys():
            replaced = replace_vowels(word)
            vowel_words.setdefault(replaced, lower_words[word])
        def check(word):
            if word in wordsset:
                return word
            low = word.lower()
            result = lower_words.get(low, "")       
            if result == "":
                replaced = replace_vowels(low)
                result = vowel_words.get(replaced, "")
            return result
        return [check(query) for query in queries]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getHint(self, secret, guess):
        bulls, cows = 0, 0
        unmatched_secret, unmatched_guess = defaultdict(int), defaultdict(int)
        for s, g in zip(secret, guess):
            if s == g:
                bulls += 1
            else:
                unmatched_secret[s] += 1
                unmatched_guess[g] += 1
        for g, count in unmatched_guess.items():
            cows += min(unmatched_secret[g], count)
        return str(bulls) + "A" + str(cows) + "B"
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findDerangement(self, n):
        MODULO = 10 ** 9 + 7
        derange, one_correct = 0, 1
        for i in range(2, n + 1):
            derange, one_correct = (derange * (i - 1) + one_correct) % MODULO, (i * derange) % MODULO
        return derange
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def toHex(self, num):
        if num == 0:        
            return "0"
        if num < 0:         
            num += 2 ** 32
        result = []
        while num != 0:
            num, digit = divmod(num, 16)
            if digit > 9:   
                result.append(chr(ord("a") + digit - 10))
            else:
                result.append(str(digit))
        return "".join(result[::-1])    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def partition(self, head, x):
        lesser_head = lesser = ListNode(None)
        greater_head = greater = ListNode(None)
        node = head
        while node:
            if node.val < x:
                lesser.next = node
                lesser = node
            else:                       
                greater.next = node
                greater = node
            node = node.next
        greater.next = None                 
        lesser.next = greater_head.next     
        return lesser_head.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def hIndex(self, citations):
        buckets = [0] * (len(citations)+1)      
        for citation in citations:              
            buckets[min(citation, len(citations))] += 1
        papers = 0                              
        for bucket in range(len(buckets)-1, -1, -1):
            papers += buckets[bucket]
            if papers >= bucket:
                return bucket
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def complexNumberMultiply(self, a, b):
        a_real, a_im = a.split("+")
        a_real, a_im = int(a_real), int(a_im[:-1])
        b_real, b_im = b.split("+")
        b_real, b_im = int(b_real), int(b_im[:-1])
        c_real = a_real * b_real - a_im * b_im
        c_im = a_real * b_im + a_im * b_real
        return str(c_real) + "+" + str(c_im) + "i"
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLeaves(self, root):
        leaves = []     
        self.height(root, leaves)
        return leaves
    def height(self, node, leaves):
        if not node:
            return -1
        h = 1 + max(self.height(node.left, leaves), self.height(node.right, leaves))
        if h >= len(leaves):    
            leaves.append([])
        leaves[h].append(node.val)
        return h
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestValidParentheses(self, s):
        stack = []                  
        for i, c in enumerate(s):
            if c == ")" and stack and s[stack[-1]] == '(':
                stack.pop()         
            else:
                stack.append(i)     
        stack.append(len(s))        
        max_length = stack[0]       
        for index in range(1, len(stack)):  
            max_length = max(max_length, stack[index] - stack[index-1] - 1)
        return max_length
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def kSmallestPairs(self, nums1, nums2, k):
        if not nums1 or not nums2:
            return []
        smallest = []
        frontier = [(nums1[0] + nums2[0], 0, 0)]
        while frontier and len(smallest) < k:
            _, i, j = heapq.heappop(frontier)
            smallest.append([nums1[i], nums2[j]])
            if len(frontier) >= k:      
                continue
            if i < len(nums1) - 1:      
                heapq.heappush(frontier, (nums1[i + 1] + nums2[j], i + 1, j))
            if i == 0 and j < len(nums2) - 1:      
                heapq.heappush(frontier, (nums1[i] + nums2[j + 1], i, j + 1))
        return smallest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def sortArrayByParityII(self, A):
        odd = 1
        for even in range(0, len(A), 2):
            if A[even] % 2 == 1:
                while A[odd] % 2 == 1:
                    odd += 2
                A[odd], A[even] = A[even], A[odd]
        return A
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        erase = 0
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x.start)   
        current_end = intervals[0].start        
        for interval in intervals:
            if current_end > interval.start:    
                erase += 1
                if interval.end > current_end:  
                    continue
            current_end = interval.end          
        return erase
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def generatePossibleNextMoves(self, s):
        result = []
        for i in range(len(s) - 1):
            if s[i:i + 2] == "++":
                result.append(s[:i] + "--" + s[i + 2:])
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minArea(self, image, x, y):
        if not image or not image[0] or image[x][y] != '1':
            return 0
        top_edge = self.find_edge(0, x, True, True, image)
        bottom_edge = self.find_edge(x+1, len(image), True, False, image)
        left_edge = self.find_edge(0, y, False, True, image)
        right_edge = self.find_edge(y+1, len(image[0]), False, False, image)
        return (right_edge - left_edge) * (bottom_edge - top_edge)
    def find_edge(self, left, right, column, black, image):
        while left < right:
            mid = (left + right) // 2
            if black == self.any_black(mid, column, image):
                right = mid
            else:
                left = mid + 1
        return left
    def any_black(self, i, column, image):
        if column:      
            return ('1' in image[i])
        else:           
            return any(image[r][i] == '1' for r in range(len(image)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maximalRectangle(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        rows = len(matrix)
        cols = len(matrix[0])
        max_area = 0
        heights = [0] * cols
        for row in range(rows):
            heights = [heights[i]+1 if matrix[row][i]=='1' else 0 for i in range(cols)]
            heights.append(0)
            stack = [0]
            for col in range(1, len(heights)):
                while stack and heights[col] < heights[stack[-1]]:
                    height = heights[stack.pop()]
                    if not stack:
                        width = col
                    else:
                        width = col - stack[-1] - 1
                    max_area = max(max_area, height * width)
                stack.append(col)
        return max_area
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minRefuelStops(self, target, startFuel, stations):
        stops = 0
        fuel = startFuel                                
        past_fuels = []                                 
        stations.append([target, 0])                    
        for distance, station_fuel in stations:
            while fuel < distance:                      
                if not past_fuels:                      
                    return -1
                fuel -= heapq.heappop(past_fuels)       
                stops += 1
            heapq.heappush(past_fuels, -station_fuel)   
        return stops
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def removeDuplicates(self, nums):
        next = 2    
        for index in range(2, len(nums)):
            if nums[index] != nums[next-2]:     
                nums[next] = nums[index]
                next += 1
        return min(next, len(nums))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def guessNumber(self, n):
        low, high = 1, n
        while True:
            mid = (low + high) // 2
            g = guess(mid)
            if g == -1:         
                high = mid - 1
            elif g == 1:        
                low = mid + 1
            else:
                return mid
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        x_lhs = max(A, E)
        x_rhs = min(C, G)
        x_overlap = max(x_rhs - x_lhs, 0)
        y_lhs = max(B, F)
        y_rhs = min(D, H)
        y_overlap = max(y_rhs - y_lhs, 0)
        rect1 = (C - A) * (D - B)
        rect2 = (G - E) * (H - F)
        return rect1 + rect2 - y_overlap * x_overlap
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxIncreaseKeepingSkyline(self, grid):
        rows, cols = len(grid), len(grid[0])
        row_sky = [0 for _ in range(rows)]          
        col_sky = [0 for _ in range(cols)]          
        for r in range(rows):
            for c in range(cols):
                row_sky[r] = max(row_sky[r], grid[r][c])
                col_sky[c] = max(col_sky[c], grid[r][c])
        increase = 0
        for r in range(rows):
            for c in range(cols):
                increase += min(row_sky[r], col_sky[c]) - grid[r][c]
        return increase
EOF
n,studentMarks = int(input()),namedtuple('sM',input().split())
print("%.2f" % (sum([float(s.MARKS) for s in [studentMarks(*input().split()) for _ in range(n)]])/n))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestPalindrome(self, n):
        if n == 1:
            return 9
        for a in range(1, 9 * 10 ** (n - 1)):
            hi = (10 ** n) - a          
            lo = int(str(hi)[::-1])     
            if a ** 2 - 4 * lo < 0:
                continue
            if (a ** 2 - 4 * lo) ** .5 == int((a ** 2 - 4 * lo) ** .5):     
                return (lo + 10 ** n * (10 ** n - a)) % 1337
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def minDepth(self, root):
        if not root:
            return 0
        l = self.minDepth(root.left)
        r = self.minDepth(root.right)
        if not l or not r:
            return 1 + l + r
        return 1 + min(l, r)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMaxAverage(self, nums, k):
        window_sum = sum(nums[:k])          
        max_average = window_sum / float(k)
        for i in range(len(nums) - k):
            window_sum += nums[i + k] - nums[i]
            max_average = max(max_average, window_sum / float(k))
        return max_average
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isMatch(self, s, p):
        i, j = 0, 0         
        star = -1           
        while i < len(s):
            if j >= len(p) or (p[j] not in {'*' , '?'} and p[j] != s[i]):
                if star == -1:      
                    return False
                j = star + 1        
                star_i += 1         
                i = star_i
            elif p[j] == '*':       
                star = j
                star_i = i
                j += 1
            else:                   
                i += 1
                j += 1
        while j < len(p):           
            if p[j] != '*':
                return False
            j += 1
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lengthLongestPath(self, input):
        longest = 0
        depths = [0]    
        for line in input.splitlines():
            stripped = line.lstrip("\t")
            depth = len(line) - len(stripped)
            if "." in line:
                longest = max(longest, len(stripped) + depth + depths[depth])
            else:
                if len(depths) <= depth + 1:
                    depths.append(0)
                depths[depth + 1] = depths[depth] + len(stripped)
        return longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def shortestSuperstring(self, A):
        N = len(A)
        overlaps = [[0] * N for _ in range(N)]  
        for i, x in enumerate(A):
            for j, y in enumerate(A):
                if i != j:
                    for ans in range(min(len(x), len(y)), 0, -1):
                        if x.endswith(y[:ans]):
                            overlaps[i][j] = ans
                            break
        dp = [[0] * N for _ in range(1 << N)]  
        parent = [[None] * N for _ in range(1 << N)]  
        for mask in range(1, 1 << N):  
            for bit in range(N):
                if (mask >> bit) & 1:  
                    prev_mask = mask ^ (1 << bit)  
                    if prev_mask == 0:
                        continue
                    for i in range(N):
                        if (prev_mask >> i) & 1:  
                            overlap = dp[prev_mask][i] + overlaps[i][bit]
                            if overlap > dp[mask][bit]:  
                                dp[mask][bit] = overlap
                                parent[mask][bit] = i
        mask = (1 << N) - 1
        i = max(range(N), key=lambda x: dp[-1][x])  
        result = [A[i]]
        used = {i}
        while True:
            mask, j = mask ^ (1 << i), parent[mask][i]  
            if j is None:
                break
            overlap = overlaps[j][i]
            prefix = A[j] if overlap == 0 else A[j][:-overlap]
            result.append(prefix)
            used.add(j)
            i = j
        result = result[::-1] + [A[i] for i in range(N) if i not in used]  
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findRedundantDirectedConnection(self, edges):
        n = len(edges)
        parents = [[] for _ in range(n + 1)]
        nbors = [set() for _ in range(n + 1)]
        for a, b in edges:                      
            parents[b].append(a)
            nbors[a].add(b)
        root = None
        for i, parent in enumerate(parents):    
            if len(parent) == 2:
                two_parents = i
            if not parent:                      
                root = i
        def valid(root):                        
            visited = set()
            queue = [root]
            while queue:
                node = queue.pop()
                if node in visited:
                    return False
                visited.add(node)
                queue += nbors[node]
            return len(visited) == n
        if root:                                
            p1, p2 = parents[two_parents]
            nbors[p2].discard(two_parents)      
            return [p2, two_parents] if valid(root) else [p1, two_parents]
        for i in range(len(edges) - 1, -1, -1): 
            n1, n2 = edges[i]
            nbors[n1].discard(n2)               
            if valid(n2):                       
                return edges[i]
            nbors[n1].add(n2)                   
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class FreqStack(object):
    def __init__(self):
        self.counter = defaultdict(int)     
        self.stack_of_stacks = []           
    def push(self, x):
        self.counter[x] += 1
        count = self.counter[x]
        if count > len(self.stack_of_stacks):   
            self.stack_of_stacks.append([])
        self.stack_of_stacks[count - 1].append(x)
    def pop(self):
        num = self.stack_of_stacks[-1].pop()    
        self.counter[num] -= 1
        if not self.stack_of_stacks[-1]:        
            self.stack_of_stacks.pop()
        return num
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def generatePalindromes(self, s):
        char_counts = Counter(s)
        odd_char = ""                   
        for char, count in char_counts.items():
            if count % 2 != 0:
                if odd_char:
                    return []               
                odd_char = char
                char_counts[odd_char] -= 1  
        palindromes = []
        self.build_palindromes(palindromes, [], char_counts, len(s)//2)
        return ["".join(p + [odd_char] + p[::-1]) for p in palindromes]
    def build_palindromes(self, palindromes, partial, char_counts, remaining):
        if remaining == 0:
            palindromes.append(partial[:])      
        for c in char_counts.keys():
            if char_counts[c] == 0:             
                continue
            char_counts[c] -= 2                 
            partial.append(c)
            self.build_palindromes(palindromes, partial, char_counts, remaining-1)
            partial.pop()                      
            char_counts[c] += 2
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def convertBST(self, root):
        self.running_sum = 0
        def inorder(node):
            if not node:
                return
            inorder(node.right)
            node.val += self.running_sum
            self.running_sum = node.val
            inorder(node.left)
        inorder(root)
        return root
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class CBTInserter:
    def __init__(self, root):
        self.nodelist = [root]
        for node in self.nodelist:          
            if node.left:
                self.nodelist.append(node.left)
            if node.right:
                self.nodelist.append(node.right)
    def insert(self, v):
        node = TreeNode(v)                  
        self.nodelist.append(node)
        n = len(self.nodelist)
        parent = self.nodelist[(n // 2) - 1]
        if n % 2 == 0:
            parent.left = node
        else:
            parent.right = node
        return parent.val
    def get_root(self):
        return self.nodelist[0]     
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minimumDeleteSum(self, s1, s2):
        dp = [0]        
        for c in s2:    
            dp.append(dp[-1] + ord(c))
        for i in range(len(s1)):
            new_dp = [dp[0] + ord(s1[i])]   
            for j in range(len(s2)):
                if s1[i] == s2[j]:          
                    new_dp.append(dp[j])
                else:                       
                    new_dp.append(min(ord(s1[i]) + dp[j + 1], ord(s2[j]) + new_dp[-1]))
            dp = new_dp
        return dp[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def cheapestJump(self, A, B):
        cheapest = [[float("inf"), []] for _ in range(len(A))]  
        cheapest[0] = [A[0], [1]]
        for i, cost in enumerate(A[:-1]):                       
            if cost == -1:                                      
                continue
            for j in range(i + 1, min(i + B + 1, len(A))):
                if A[j] == -1:                                  
                    continue
                new_cost = cheapest[i][0] + A[j]
                new_path = cheapest[i][1] + [j + 1]             
                cheapest[j] = min(cheapest[j], [new_cost, new_path])    
        return cheapest[-1][1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numDistinctIslands(self, grid):
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        def BFS(r, c):
            queue = [(0, 0)]  
            for r_rel, c_rel in queue:  
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_r, new_c = r_rel + dr + r, c_rel + dc + c
                    if 0 <= new_r < rows and 0 <= new_c < cols and grid[new_r][new_c] == 1:
                        grid[new_r][new_c] = 0
                        queue.append((new_r - r, new_c - c))
            return tuple(queue)
        islands = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    continue
                islands.add(BFS(r, c))
        return len(islands)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkSubarraySum(self, nums, k):
        prefix_sum, prefix_sums = 0, {0: -1}  
        for i, n in enumerate(nums):
            prefix_sum += n
            if k != 0:  
                prefix_sum = prefix_sum % k
            if prefix_sum in prefix_sums:
                if i - prefix_sums[prefix_sum] > 1:     
                    return True
            else:
                prefix_sums[prefix_sum] = i  
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def convert(self, s, numRows):
        if numRows == 1:
            return s
        zigzag = [[] for _ in range(numRows)]
        row = 0
        direction = -1      
        for c in s:
            zigzag[row].append(c)
            if row == 0 or row == numRows-1:    
                direction = -direction
            row += direction
        return "".join([c for r in zigzag for c in r])  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findAndReplacePattern(self, words, pattern):
        def canonical(s):               
            result = []                 
            mapping = {}                
            value = 0                   
            for c in s:
                if c not in mapping:    
                    mapping[c] = value
                    value += 1
                result.append(mapping[c])
            return tuple(result)
        pattern = canonical(pattern)
        return [word for word in words if canonical(word) == pattern]
EOF
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        directions = [(-1, -1), (-1, 0), (-1, 1), \
                      ( 0, -1), ( 0, 1), \
                      ( 1, -1), ( 1, 0), ( 1, 1)]
        result = 0
        q = collections.deque([(0, 0)])
        while q:
            result += 1
            next_depth = collections.deque()
            while q:
                i, j = q.popleft()
                if 0 <= i < len(grid) and \
                   0 <= j < len(grid[0]) and \
                    not grid[i][j]:
                    grid[i][j] = 1
                    if i == len(grid)-1 and j == len(grid)-1:
                        return result
                    for d in directions:
                        next_depth.append((i+d[0], j+d[1]))
            q = next_depth
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def totalFruit(self, tree):
        prev = [None, float("inf"), float("inf")]   
        other = [None, float("inf"), float("inf")]
        result = 1
        for i, fruit in enumerate(tree):
            if fruit == prev[0]:
                prev[2] = i
                result = max(result, i + 1 - min(prev[1], other[1]))
            elif fruit == other[0]:
                other[2] = i
                other, prev = prev, other
                result = max(result, i + 1 - min(prev[1], other[1]))
            elif prev[0] is None:
                prev = [fruit, i, i]
            elif other[0] is None:
                other, prev = prev, [fruit, i, i]
                result = max(result, i + 1 - other[1])
            else:
                other = [prev[0], other[2] + 1, prev[2]]
                prev = [fruit, i, i]
        return result
EOF
arr = []
def generate(arr, k):
    for el in combinations_with_replacement(arr, k):
        print(el, end=' ')
    print()
    for el in combinations_with_replacement(arr, k):
        print(sum(el), end=' ')
def delete_red(deep, output, ind, cur):
    global arr
    if ind == 0:
        arr.remove(cur)
    else:
        while deep >= 0:
            delete_red(deep, output, ind-1, cur+output[deep])
            deep -= 1
def solution(n, k):
    global arr
    output = [arr[0]//k]
    del arr[0]
    for ind in range(1, n):
        el = arr[0] - (k-1)*output[0]
        output.append(el)
        if ind < n-1:
            delete_red(ind, output, k-1, output[-1])
    return output
if __name__ == "__main__":
    t = int(input().strip())
    for _ in range(t):
        n, k = [int(x) for x in input().strip().split()]
        arr = [int(x) for x in input().strip().split()]
        arr = sorted(arr)
        print(" ".join(map(str, solution(n, k))))
EOF
def find_number(n):
    three = n//3
    while (n - 3*three)%5 != 0:
        three -= 1
    if three > 0:
        res = [5] * 3 * three
        res += [3] * (n - 3*three)
    elif three == 0 and n%5 == 0:
        res = [3] * n
    else:
        res = [-1]
    return res
t = int(input().strip())
for a0 in range(t):
    n = int(input().strip())
    print("".join(map(str, find_number(n))))
EOF
def appendAndDelete(s, t, k):
    start = 0
    ind = 0
    to_del = 0
    to_app = 0
    while ind < len(s) and ind < len(t) and s[ind] == t[ind]:
        ind += 1
    start = ind
    if start < len(s):
        to_del = len(s[start:])
        if to_del == len(s) and k - to_del >= len(t):
            return 'Yes'
    if start < len(t):
        to_app = len(t[start:])
    k -= to_del + to_app
    if k == 0 or (k > 0 and k % 2 == 0) or k >= 2*len(t):
        return 'Yes'
    else:
        return 'No'
if __name__ == "__main__":
    s = input().strip()
    t = input().strip()
    k = int(input().strip())
    result = appendAndDelete(s, t, k)
    print(result)
EOF
def is_safe(grid, x, y, distances):
    return x >= 0 and x < len(grid) and y >= 0 and y < len(grid) and distances[x][y] == -1 and grid[x][y] != 'X' 
def get_safe_moves(grid, node, distances):
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    variants = []
    for di in directions:
        nunode = (node[0] + di[0], node[1] + di[1])
        while is_safe(grid, nunode[0], nunode[1], distances):
            variants.append(nunode)
            nunode = (nunode[0] + di[0], nunode[1] + di[1])
    return variants
def minimumMoves(grid, startX, startY, goalX, goalY):
    next_to_visit = deque()
    node = (startX, startY)
    next_to_visit.appendleft(node)
    distances = [[-1]*len(grid) for _ in range(len(grid))]
    distances[startX][startY] = 0
    while next_to_visit:
        node = next_to_visit.pop()
        height = distances[node[0]][node[1]]
        variants = get_safe_moves(grid, node, distances)
        for var in variants:
            if var == (goalX, goalY):
                return height + 1
            distances[var[0]][var[1]] = height + 1
            next_to_visit.appendleft(var)
    return -1
if __name__ == "__main__":
    n = int(input().strip())
    grid = []
    for _ in range(n):
        layer = list(input().strip())
        grid.append(layer)
    startX, startY, goalX, goalY = [int(i) for i in input().strip().split()]
    result = minimumMoves(grid, startX, startY, goalX, goalY)
    print(result)
EOF
alpha = string.ascii_lowercase
def funnyString(s):
    res = 'Funny'
    r = s[::-1]
    for ind in range(1, len(s)):
        if abs(alpha.index(s[ind]) - alpha.index(s[ind-1])) != abs(alpha.index(r[ind]) - alpha.index(r[ind-1])):
            res = 'Not Funny'
            break
    return res
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = funnyString(s)
    print(result)
EOF
def miniMaxSum(arr):
    sum_min = math.inf
    sum_max = -math.inf
    sum_temp = 0
    for exclude in range(len(arr)):
        sum_temp = 0
        for ind, elem in enumerate(arr):
            if ind == exclude:
                continue
            sum_temp += elem
        if sum_temp < sum_min:
            sum_min = sum_temp
        if sum_temp > sum_max:
            sum_max = sum_temp
    print("{} {}".format(sum_min, sum_max))
    return sum_min, sum_max
if __name__ == "__main__":
    arr = list(map(int, input().strip().split(' ')))
    miniMaxSum(arr)
EOF
n = int(input().strip())
a = list(map(int, input().strip().split(' ')))
def bub_sort(arr, n):
    num_swaps = 0
    for i in range(n):
        for j in range(n-1):
            if arr[j] > a[j+1]:
                buff = arr[j+1]
                arr[j+1] = a[j]
                a[j] = buff
                num_swaps += 1
    print("Array is sorted in {} swaps.".format(num_swaps))
    print("First Element: {}".format(arr[0]))
    print("Last Element: {}".format(arr[-1]))
bub_sort(a, n)
EOF
def cavityMap(grid):
    leng = len(grid)
    output = [[0]*leng for _ in range(leng)]
    for ind in range(leng):
        for jnd in range(leng):
            if (0 < ind < leng-1 and 0 < jnd < leng-1 and
               grid[ind][jnd] > grid[ind+1][jnd] and
               grid[ind][jnd] > grid[ind-1][jnd] and
               grid[ind][jnd] > grid[ind][jnd+1] and
               grid[ind][jnd] > grid[ind][jnd-1]):
                output[ind][jnd] = 'X'
            else:
                output[ind][jnd] = grid[ind][jnd]
    return output
if __name__ == "__main__":
    n = int(input().strip())
    grid = []
    grid_i = 0
    for grid_i in range(n):
        grid_t = list(map(int, input().strip()))
        grid.append(grid_t)
    result = cavityMap(grid)
    print("\n".join(["".join(map(str, x)) for x in result]))
EOF
def closestNumbers(arr):
    output = []
    arr = sorted(arr)
    nowmin = 10**9
    for ind in range(1, len(arr)):
        diff = abs(arr[ind-1] - arr[ind])
        if diff < nowmin:
            output = [(arr[ind-1], arr[ind])]
            nowmin = diff
        elif diff == nowmin:
            output.append((arr[ind-1], arr[ind]))
    flat_list = [item for sublist in output for item in sublist]
    return flat_list
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = closestNumbers(arr)
    print (" ".join(map(str, result)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestTimeFromDigits(self, A):
        best_minutes = -1       
        for time in permutations(A):
            hours = time[0] * 10 + time[1]
            if hours >= 24:
                continue
            minutes = time[2] * 10 + time[3]
            if minutes >= 60:
                continue
            total_minutes = hours * 60 + minutes
            if total_minutes > best_minutes:
                best_minutes = total_minutes
                best = time
        if best_minutes == -1:
            return ""
        best = [str(x) for x in best]
        return "".join(best[:2]) + ":" + "".join(best[2:])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def subarrayBitwiseORs(self, A):
        all_or, subarray_or = set(), set()
        for num in A:
            new_or = {num | x for x in subarray_or}
            new_or.add(num)
            all_or |= new_or
            subarray_or = new_or
        return len(all_or)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findBottomLeftValue(self, root):
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)
        return node.val
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxRotateFunction(self, A):
        rotate_val, sum_A = 0, 0
        for i, num in enumerate(A):
            sum_A += num
            rotate_val += i * num
        max_rotate = rotate_val
        for i in range(len(A) - 1, -1, -1):
            rotate_val += sum_A - (len(A) * A[i])       
            max_rotate = max(max_rotate, rotate_val)
        return max_rotate
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def insert(self, intervals, newInterval):
        left, right = 0, len(intervals)-1
        while left < len(intervals) and intervals[left].end < newInterval.start:
            left += 1
        while right >= 0 and intervals[right].start > newInterval.end:
            right -= 1
        if left <= right:
            newInterval.start = min(newInterval.start, intervals[left].start)
            newInterval.end = max(newInterval.end, intervals[right].end)
        return intervals[:left] + [newInterval] + intervals[right+1:]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def calculate(self, s):
        operators = {"+", "-", "*", "/"}
        def parse(i):                                           
            parsed = []
            while i < len(s):
                c = s[i]
                if c in operators:
                    parsed.append(c)
                elif "0" <= c <= "9":
                    if parsed and isinstance(parsed[-1], int):  
                        parsed[-1] = parsed[-1] * 10 + int(c)
                    else:                                       
                        parsed.append(int(c))
                elif c == "(":
                    sublist, i = parse(i + 1)
                    parsed.append(sublist)
                elif c == ")":
                    return parsed, i                            
                i += 1                                          
            return parsed, len(s)
        def calculate(tokens):
            if isinstance(tokens, int):                         
                return tokens
            result = [calculate(tokens[0])]                     
            for i in range(1, len(tokens), 2):                  
                op, num = tokens[i], calculate(tokens[i + 1])
                if op == "/":
                    result.append(result.pop() // num)
                elif op == "*":
                    result.append(result.pop() * num)
                elif op == "+":
                    result.append(num)
                else:
                    result.append(-num)
            return sum(result)
        parsed_s, _ = parse(0)
        return calculate(parsed_s)
EOF
if __name__ == '__main__':
    n = int(input())
    integer_list = tuple((map(int, input().split())))
    print(hash(integer_list))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class NumArray(object):
    def __init__(self, nums):
        self.width = int(len(nums)**0.5)    
        self.bin_sums = []                  
        self.nums = nums
        for i, num in enumerate(nums):
            if i % self.width == 0:         
                self.bin_sums.append(num)
            else:                           
                self.bin_sums[-1] += num
    def update(self, i, val):
        bin_i = i // self.width
        diff = val - self.nums[i]
        self.bin_sums[bin_i] += diff        
        self.nums[i] = val                  
    def sumRange(self, i, j):
        bin_i, bin_j = i // self.width, j // self.width
        range_sum = sum(self.bin_sums[bin_i:bin_j])         
        range_sum += sum(self.nums[bin_j*self.width:j+1])   
        range_sum -= sum(self.nums[bin_i*self.width:i])     
        return range_sum
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def generateTrees(self, n):
        if n <= 0:
            return []
        return self.generate(1, n)
    def generate(self, left, right):
        if left > right:
            return [None]
        results = []
        for i in range(left, right+1):
            left_trees = self.generate(left, i-1)
            right_trees = self.generate(i+1, right)
            for l in left_trees:
                for r in right_trees:
                    root = TreeNode(i)
                    root.left = l
                    root.right = r
                    results.append(root)
        return results
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findReplaceString(self, S, indexes, sources, targets):
        replaced = [c for c in S]               
        for i, src, tgt in zip(indexes, sources, targets):
            n = len(src)
            if S[i:i + n] == src:
                replaced[i] = tgt
                replaced[i + 1:i + n] = [""] * (n - 1)
        return "".join(replaced)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sortTransformedArray(self, nums, a, b, c):
        def transform(x):
            return a * x * x + b * x + c
        transformed = [transform(num) for num in nums]
        left, right = 0, len(nums) - 1
        result = []
        while left <= right:
            if (a > 0 and transformed[left] > transformed[right]) or (a <= 0 and transformed[right] > transformed[left]):
                result.append(transformed[left])
                left += 1
            else:
                result.append(transformed[right])
                right -= 1
        return result[::-1] if a > 0 else result
EOF
    return sum(ar)
if __name__ == '__main__':
    f = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = aVeryBigSum(n, ar)
    f.write(str(result) + '\n')
    f.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyCalendarThree(object):
    def __init__(self):
        self.bookings = [[float("-inf"), 0], [float("inf"), 0]]     
        self.max_booking = 0                                        
    def book(self, start, end):
        i = bisect.bisect_left(self.bookings, [start, -1])          
        if self.bookings[i][0] != start:                            
            count = self.bookings[i - 1][1]
            self.bookings.insert(i, [start, count + 1])
            self.max_booking = max(self.max_booking, count + 1)
            i += 1
        while end > self.bookings[i][0]:                            
            self.bookings[i][1] += 1
            self.max_booking = max(self.max_booking, self.bookings[i][1])
            i += 1
        if self.bookings[i][0] != end:                              
            count = self.bookings[i - 1][1]
            self.bookings.insert(i, [end, count - 1])
        return self.max_booking
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findSecretWord(self, wordlist, master):
        def pair_matches(a, b):         
            return sum(c1 == c2 for c1, c2 in zip(a, b))
        def most_overlap_word():
            counts = [collections.defaultdict(int) for _ in range(6)]   
            for word in candidates:
                for i, c in enumerate(word):
                    counts[i][c] += 1                               
            return max(candidates, key=lambda x:sum(counts[i][c] for i, c in enumerate(x)))
        candidates = wordlist[:]        
        while candidates:
            s = most_overlap_word()     
            matches = master.guess(s)
            if matches == 6:
                return
            candidates = [w for w in candidates if pair_matches(s, w) == matches]   
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def letterCasePermutation(self, S):
        permutations = [[]]
        for c in S:
            if "0" <= c <= "9":
                for perm in permutations:
                    perm.append(c)
            else:
                new_permutations = []
                upper, lower = c.upper(), c.lower()
                for perm in permutations:
                    new_permutations.append(perm + [upper])
                    perm.append(lower)
                    new_permutations.append(perm)
                permutations = new_permutations
        return ["".join(perm) for perm in permutations]     
EOF
d = {}
for _ in range(int(input())):
    n,p = input().split()
    d[n] = p
while True:
    try:
        n = input()
        if n in d:
            print(n+"="+d[n])
        else:
            print("Not found")
    except:
        break
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def monotoneIncreasingDigits(self, N):
        s = [int(c) for c in str(N)]                    
        i = 0
        while i + 1 < len(s):
            if s[i + 1] < s[i]:                         
                while i > 0 and s[i] - 1 < s[i - 1]:    
                    i -= 1
                s[i] -= 1                               
                s[i + 1:] = [9] * (len(s) - i - 1)      
                result = 0
                for val in s:
                    result = result * 10 + val
                return result
            else:
                i += 1
        return N                                        
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sumSubseqWidths(self, A):
        result = 0
        n = len(A)
        A.sort()
        for i, num in enumerate(A):
            result += (1 << i) * num
            result -= (1 << (n - 1 - i)) * num
        return result % (10 ** 9 + 7)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def splitListToParts(self, root, k):
        node, count = root, 0
        while node:                 
            count += 1
            node = node.next
        part_length, odd_parts = divmod(count, k)
        result = []
        prev, node = None, root
        for _ in range(k):
            required = part_length  
            if odd_parts > 0:
                odd_parts -= 1
                required += 1
            result.append(node)
            for _ in range(required):
                prev, node = node, node.next
            if prev:                
                prev.next = None
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canJump(self, nums):
        max_index = 0
        for i, num in enumerate(nums):
            if i > max_index:
                return False
            max_index = max(max_index, i + num)
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def optimalDivision(self, nums):
        nums = [str(s) for s in nums]
        result = nums[0]
        if len(nums) == 1:
            return result
        if len(nums) == 2:
            return result + "/" + nums[1]
        return result + "/(" + "/".join(nums[1:]) + ")"
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def __init__(self):
        self.i = 0          
    def str2tree(self, s):
        def next_num():  
            num, neg = 0, False
            if s[self.i] == "-":
                neg = True
                self.i += 1
            while self.i < len(s) and s[self.i] not in {"(", ")"}:
                num = num * 10 + int(s[self.i])
                self.i += 1
            return TreeNode(-num) if neg else TreeNode(num)
        def helper():
            if self.i >= len(s):
                return None
            root = next_num()
            if self.i < len(s) and s[self.i] == "(":
                self.i += 1     
                root.left = helper()
                self.i += 1     
            if self.i < len(s) and s[self.i] == "(":
                self.i += 1     
                root.right = helper()
                self.i += 1     
            return root
        return helper()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def letterCombinations(self, digits):
        if not digits or '0' in digits or '1' in digits:
            return []
        results = [[]]
        mapping = {'2' : ['a', 'b', 'c'],
                   '3' : ['d', 'e', 'f'],
                   '4' : ['g', 'h', 'i'],
                   '5' : ['j', 'k', 'l'],
                   '6' : ['m', 'n', 'o'],
                   '7' : ['p', 'q', 'r', 's'],
                   '8' : ['t', 'u', 'v'],
                   '9' : ['w', 'x', 'y' , 'z']}
        for digit in digits:
            temp = []
            for result in results:
                for letter in mapping[digit]:
                    temp.append(result + [letter])
            results = temp
        return ["".join(result) for result in results]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def hasGroupsSizeX(self, deck):
        freq = Counter(deck)
        min_count = min(freq.values())
        if min_count == 1:              
            return False
        for X in range(2, min_count + 1):
            if all(count % X == 0 for count in freq.values()):
                return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isIdealPermutation(self, A):
        max_before_prev = -1
        for i in range(1, len(A)):
            if A[i] < max_before_prev:
                return False
            max_before_prev = max(max_before_prev, A[i - 1])
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minmaxGasDist(self, stations, K):
        distances = [s1 - s2 for s1, s2 in zip(stations[1:], stations)]
        distances.sort(reverse=True)
        def can_minmax_dist(d):
            remaining = K
            for dist in distances:
                if dist < d or remaining < 0:   
                    break
                remaining -= int(dist / d)      
            return remaining >= 0
        max_d, min_d = distances[0], 0          
        while max_d - min_d > 10 ** -6:
            mid = (max_d + min_d) / 2.0
            if can_minmax_dist(mid):
                max_d = mid                     
            else:
                min_d = mid                     
        return max_d
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def medianSlidingWindow(self, nums, k):
        lower, upper = [], []           
        for i in range(k):              
            heapq.heappush(upper, nums[i])
        for i in range(k // 2):         
            heapq.heappush(lower, -heapq.heappop(upper))
        medians = []
        junk = defaultdict(int)  
        for i in range(k, len(nums)):
            if k % 2 == 1:
                medians.append(float(upper[0]))
            else:
                medians.append((upper[0] - lower[0]) / 2.0)
            balance = 0     
            if nums[i - k] >= upper[0]:
                balance -= 1
            else:
                balance += 1
            junk[nums[i - k]] += 1
            if nums[i] >= upper[0]:
                balance += 1
                heapq.heappush(upper, nums[i])
            else:
                balance -= 1
                heapq.heappush(lower, -nums[i])
            if balance > 0:
                heapq.heappush(lower, -heapq.heappop(upper))
            elif balance < 0:
                heapq.heappush(upper, -heapq.heappop(lower))
            while upper and upper[0] in junk:
                removed = heapq.heappop(upper)
                junk[removed] -= 1
                if junk[removed] == 0:
                    del junk[removed]
            while lower and -lower[0] in junk:
                removed = -heapq.heappop(lower)
                junk[removed] -= 1
                if junk[removed] == 0:
                    del junk[removed]
        if k % 2 == 1:
            medians.append(float(upper[0]))
        else:
            medians.append((upper[0] - lower[0]) / 2.0)
        return medians
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def levelOrder(self, root):
        if not root:
            return []
        result = []
        level = [root]      
        while level:
            new_level = []
            for node in level:
                new_level += node.children
            result.append([node.val for node in level])
            level = new_level
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def intersection(self, nums1, nums2):
        return list(set(nums1) & set(nums2))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def predictPartyVictory(self, senate):
        n = len(senate)
        d, r = deque(), deque()         
        for i, c in enumerate(senate):
            if c == "D":
                d.append(i)
            else:
                r.append(i)
        while d and r:
            d_senator = d.popleft()
            r_senator = r.popleft()
            if d_senator < r_senator:
                d.append(d_senator + n)
            else:
                r.append(r_senator + n)
        return "Radiant" if r else "Dire"
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def similarRGB(self, color):
        result = ["
        for i in range(1, 6, 2):            
            first, second = int(color[i], 16), int(color[i + 1], 16)    
            difference = first - second
            if abs(difference) <= 8:
                char = color[i]
            elif difference > 0:
                char = hex(first - 1)[2]    
            else:                           
                char = hex(first + 1)[2]
            result.append(char * 2)
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def sortList(self, head):
        if not head or not head.next:
            return head
        fast, slow, prev = head, head, None
        while fast and fast.next:
            prev, slow, fast = slow, slow.next, fast.next.next
        prev.next = None
        one = self.sortList(head)
        two = self.sortList(slow)
        return self.merge(one, two)
    def merge(self, one, two):
        dummy = merged = ListNode(None)
        while one and two:
            if one.val <= two.val:
                merged.next = one
                one = one.next
            else:
                merged.next = two
                two = two.next
            merged = merged.next
        merged.next = one or two    
        return dummy.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lexicalOrder(self, n):
        lexical = [1]
        while len(lexical) < n:
            num = lexical[-1] * 10
            while num > n:      
                num //= 10
                num += 1
                while num % 10 == 0:    
                    num //= 10
            lexical.append(num)
        return lexical
class Solution2(object):
    def lexicalOrder(self, n):
        strings = list(map(str, range(1, n + 1)))
        strings.sort()
        return list(map(int, strings))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reverse(self, x):
        negative = x < 0    
        x = abs(x)
        reversed = 0
        while x != 0:
            reversed = reversed * 10 + x % 10
            x //= 10
        if reversed > 2**31 - 1:    
            return 0
        return reversed if not negative else -reversed
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def employeeFreeTime(self, schedule):
        employees = len(schedule)
        next_start_times = [(schedule[i][0].start, 0, i) for i in range(employees)]
        heapq.heapify(next_start_times)             
        last_end_time = next_start_times[0][0]      
        result = []
        while next_start_times:
            interval_start_time, interval, employee = heapq.heappop(next_start_times)
            if interval + 1 < len(schedule[employee]):  
                heapq.heappush(next_start_times, (schedule[employee][interval + 1].start, interval + 1, employee))
            if interval_start_time > last_end_time:     
                result.append(Interval(last_end_time, interval_start_time))
            last_end_time = max(last_end_time, schedule[employee][interval].end)    
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minKBitFlips(self, A, K):
        flips = 0
        flip_i = deque()            
        for i in range(len(A)):
            while flip_i and flip_i[0] + K <= i:    
                flip_i.popleft()
            if (A[i] + len(flip_i)) % 2 == 0:       
                if i > len(A) - K:
                    return -1                       
                flips += 1
                flip_i.append(i)
        return flips
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def decodeString(self, s):
        stack = []
        repeats = 0
        digits = set("0123456789")
        for c in s:
            if c == "]":
                item = stack.pop()
                current = []
                while not isinstance(item, int):
                    current.append(item)
                    item = stack.pop()
                stack += (current[::-1] * item)
            elif c in digits:
                repeats = repeats * 10 + int(c)
            elif c == "[":              
                stack.append(repeats)
                repeats = 0
            else:
                stack.append(c)
        return "".join(stack)
class Solution2(object):
    def decodeString(self, s):
        self.i = 0  
        return "".join(self.decode(s))
    def decode(self, s):
        result = []
        while self.i < len(s) and s[self.i] != "]":     
            if s[self.i] not in "0123456789":           
                result.append(s[self.i])
                self.i += 1
            else:
                repeats = 0
                while s[self.i] in "0123456789":        
                    repeats = repeats * 10 + int(s[self.i])
                    self.i += 1
                self.i += 1                             
                result += (self.decode(s) * repeats)    
                self.i += 1                             
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def mirrorReflection(self, p, q):
        k = 1
        while (k * q) % p != 0:         
            k += 1
        if k % 2 == 0:                  
            return 2
        if ((k * q) // p) % 2 == 0:     
            return 0
        return 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestSubstring(self, s, k):
        longest = 0
        to_split = [s]              
        while to_split:
            t = to_split.pop()      
            freq = Counter(t)
            splitted = [t]          
            for c in freq:
                if freq[c] < k:     
                    new_splitted = []
                    for spl in splitted:
                        new_splitted += spl.split(c)
                    splitted = new_splitted
            if len(splitted) == 1:  
                longest = max(longest, len(splitted[0]))
            else:                   
                to_split += [spl for spl in splitted if len(spl) > longest]
        return longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def numPermsDISequence(self, S):
        dp = [1] * (len(S) + 1)     
        for move in S:
            if move == "D":
                dp = dp[1:]         
                for i in range(len(dp) - 2, -1, -1):
                    dp[i] += dp[i + 1]      
            else:
                dp = dp[:-1]        
                for i in range(1, len(dp)):
                    dp[i] += dp[i - 1]      
        return dp[0] % (10 ** 9 + 7)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def intersect(self, nums1, nums2):
        freq1 = Counter(nums1)
        result = []
        for i, num in enumerate(nums2):
            if num in freq1 and freq1[num] > 0:
                freq1[num] -= 1
                result.append(num)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def containsDuplicate(self, nums):
        return len(set(nums)) != len(nums)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def pivotIndex(self, nums):
        left, right = 0, sum(nums)          
        for i, num in enumerate(nums):
            right -= num
            if left == right:
                return i
            left += num                     
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestSumOfAverages(self, A, K):
        memo = {}
        def helper(i, k):                   
            if (i, k) in memo:
                return memo[(i, k)]
            if k == 1:
                memo[(i, k)] = sum(A[:i]) / float(i)
                return memo[(i, k)]
            best = 0
            for j in range(k - 1, i):       
                best = max(best, helper(j, k - 1) + sum(A[j:i]) / float(i - j))
            memo[(i, k)] = best
            return best
        return helper(len(A), K)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lemonadeChange(self, bills):
        fives, tens = 0, 0          
        for bill in bills:
            if bill == 5:           
                fives += 1
            elif bill == 10:        
                if fives == 0:
                    return False
                fives -= 1
                tens += 1
            elif bill == 20:        
                if tens >= 1 and fives >= 1:
                    tens -= 1
                    fives -= 1
                elif fives >= 3:    
                    fives -= 3
                else:
                    return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findShortestSubArray(self, nums):
        counts, limits = defaultdict(int), {}
        for i, num in enumerate(nums):
            counts[num] += 1
            if num not in limits:
                limits[num] = [i, i]    
            else:
                limits[num][-1] = i     
        max_count, max_nums = 0, []     
        for num, count in counts.items():
            if count == max_count:
                max_nums.append(num)    
            elif count > max_count:
                max_nums = [num]        
                max_count = count
        shortest = float("inf")
        for num in max_nums:
            shortest = min(shortest, limits[num][1] - limits[num][0] + 1)
        return shortest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def surfaceArea(self, grid):
        n = len(grid)
        area = 0
        for row in range(n):
            for col in range(n):
                if grid[row][col] == 0:         
                    continue
                height = grid[row][col]
                area += 4 * height + 2
                if row != 0:
                    area -= min(grid[row - 1][col], height)
                if col != 0:
                    area -= min(grid[row][col - 1], height)
                if row != n - 1:
                    area -= min(grid[row + 1][col], height)
                if col != n - 1:
                    area -= min(grid[row][col + 1], height)
        return area
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def escapeGhosts(self, ghosts, target):
        def manhattan(position):
            return abs(position[0] - target[0]) + abs(position[1] - target[1])
        target_distance = manhattan((0, 0))
        return all(manhattan(ghost) > target_distance for ghost in ghosts)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkValidString(self, s):
        min_open, max_open = 0, 0
        for c in s:
            if c == "(":
                min_open += 1
                max_open += 1
            elif c == ")":
                min_open = max(0, min_open - 1)
                max_open -= 1
            else:
                min_open = max(0, min_open - 1)
                max_open += 1
            if max_open < 0:
                return False
        return min_open == 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findComplement(self, num):
        i = 1
        while i <= num:
            i <<= 1
        return (i - 1) ^ num
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def countNodes(self, root):
        if not root:
            return 0
        left_subtree = self.left_depth(root.left)
        right_subtree = self.left_depth(root.right)
        if left_subtree == right_subtree:
            return 2**left_subtree + self.countNodes(root.right)    
        else:
            return 2**right_subtree + self.countNodes(root.left)
    def left_depth(self, node):
        depth = 0
        while node:
            node = node.left
            depth += 1
        return depth
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TicTacToe(object):
    def __init__(self, n):
        self.rows, self.cols = [0 for _ in range(n)], [0 for _ in range(n)]
        self.d_up, self.d_down = 0, 0
    def move(self, row, col, player):
        n = len(self.rows)
        score = (2 * player) - 3    
        self.rows[row] += score
        self.cols[col] += score
        if abs(self.rows[row]) == n or abs(self.cols[col]) == n:
            return player
        if row == col:
            self.d_up += score
            if abs(self.d_up) == n:
                return player
        if row + col == n - 1:
            self.d_down += score
            if abs(self.d_down) == n:
                return player
        return 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findShortestWay(self, maze, ball, hole):
        def maze_cell(r, c):
            if [r, c] == hole:
                return -1
            elif 0 <= r < len(maze) and 0 <= c < len(maze[0]) and maze[r][c] == 0:
                return 0
            return 1
        def vertical(dirn):
            return dirn in {"d", "u"}
        def perpendicular(dirn):
            return ["r", "l"] if vertical(dirn) else ["u", "d"]     
        visited = set()  
        queue = deque()
        dirns = {"d": (1, 0), "u": (-1, 0), "r": (0, 1), "l": (0, -1)}
        for dirn in "dlru":
            queue.append((ball[0], ball[1], [dirn]))
        while queue:
            r, c, moves = queue.popleft()
            if (r, c, moves[-1]) in visited:
                continue
            visited.add((r, c, moves[-1]))
            dr, dc = dirns[moves[-1]]
            nm = maze_cell(r + dr, c + dc)
            if nm == -1:
                return "".join(moves)
            elif nm == 0:
                queue.append((r + dr, c + dc, moves))   
            elif [r, c] != ball:
                trial_dirns = perpendicular(moves[-1])
                for trial_dirn in trial_dirns:
                    queue.appendleft((r, c, moves + [trial_dirn]))  
        return "impossible"
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def threeSum(self, nums):
        results = []
        nums.sort()
        i = 0
        while i < len(nums):
            j = i + 1
            k = len(nums) - 1
            while j < k:
                triple_sum = nums[i] + nums[j] + nums[k]
                if triple_sum == 0:     
                    results.append([nums[i], nums[j], nums[k]])
                    k -= 1
                    while k > j and nums[k] == nums[k + 1]:
                        k -= 1
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                elif triple_sum > 0:    
                    k -= 1
                    while k > j and nums[k] == nums[k + 1]:
                        k -= 1
                else:                   
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
            i += 1                      
            while i < len(nums) - 2 and nums[i] == nums[i - 1]:
                i += 1
        return results
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def nextGreaterElements(self, nums):
        n = len(nums)
        stack = []                          
        next_greater = [-1] * n             
        for i in range(2 * n):
            num = nums[i % n]               
            while stack and num > nums[stack[-1]]:
                next_greater[stack.pop()] = num
            if i < n:
                stack.append(i)
        return next_greater
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMedianSortedArrays(self, A, B):
        def get_kth_smallest(a_start, b_start, k):
            if k <= 0 or k > len(A) - a_start + len(B) - b_start:
                raise ValueError('k is out of the bounds of the input lists')
            if a_start >= len(A):
                return B[b_start + k - 1]
            if b_start >= len(B):
                return A[a_start + k - 1]
            if k == 1:
                return min(A[a_start], B[b_start])
            mid_A, mid_B = float('inf'), float('inf')
            if k // 2 - 1 < len(A) - a_start:
                mid_A = A[a_start + k // 2 - 1]
            if k // 2 - 1 < len(B) - b_start:
                mid_B = B[b_start + k // 2 - 1]
            if mid_A < mid_B:
                return get_kth_smallest(a_start + k // 2, b_start, k - k // 2)
            return get_kth_smallest(a_start, b_start + k // 2, k - k // 2)
        right = get_kth_smallest(0, 0, 1 + (len(A) + len(B)) // 2)
        if (len(A) + len(B)) % 2 == 1:  
            return right
        left = get_kth_smallest(0, 0, (len(A) + len(B)) // 2)
        return (left + right) / 2.0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def repeatedSubstringPattern(self, s):
        return s in (s[1:] + s[:-1])
EOF
if __name__ == '__main__':
    n = int(input())
    for i in range(1,11):
        print(n,"x",i,"=",n*i)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestToChar(self, S, C):
        shortest = []
        prev_C = float("-inf")          
        for i, c in enumerate(S):
            if c == C:
                prev_C = i
            shortest.append(i - prev_C)
        next_C = float("inf")
        for i in range(len(S) - 1, -1, -1):
            c = S[i]
            if c == C:
                next_C = i
            shortest[i] = min(shortest[i], next_C - i)
        return shortest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def deleteDuplicates(self, head):
        node = head
        while node and node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return head
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minCut(self, s):
        min_cuts = [i-1 for i in range(len(s)+1)]       
        for i in range(len(s)):                         
            left, right = i, i                          
            while left >= 0 and right < len(s) and s[left] == s[right]:
                min_cuts[right + 1] = min(min_cuts[right + 1], 1 + min_cuts[left])
                left -= 1
                right += 1
            left, right = i, i+1                        
            while left >= 0 and right < len(s) and s[left] == s[right]:
                min_cuts[right + 1] = min(min_cuts[right + 1], 1 + min_cuts[left])
                left -= 1
                right += 1
        return min_cuts[-1]
EOF
n = int(input().strip())
for i in range(n):
    str = input()
    even_string = ''
    odd_string = ''
    for i in range(len(str)):
        if i % 2 == 0:
            even_string += str[i]
        else:
            odd_string += str[i]
    print(even_string + " " + odd_string)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def multiply(self, num1, num2):
        num1, num2 = num1[::-1], num2[::-1]         
        result = [0] * (len(num1) + len(num2))
        for i in range(len(num1)):
            int1 = ord(num1[i]) - ord('0')
            for j in range(len(num2)):
                int2 = ord(num2[j]) - ord('0')
                tens, units = divmod(int1 * int2, 10)
                result[i + j] += units      
                if result[i + j] > 9:
                    result[i + j + 1] += result[i + j] // 10
                    result[i + j] %= 10
                result[i + j + 1] += tens   
                if result[i + j + 1] > 9:
                    result[i + j + 2] += result[i + j + 1] // 10
                    result[i + j + 1] %= 10
        while len(result) > 1 and result[-1] == 0:  
            result.pop()
        return "".join(map(str, result[::-1]))      
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def groupStrings(self, strings):
        shifted = defaultdict(list)
        for s in strings:
            shift = ord(s[0]) - ord('a')            
            s_shifted = "".join([chr((ord(c) - ord('a') - shift) % 26 + ord('a')) for c in s])
            shifted[s_shifted].append(s)
        return shifted.values()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class WordFilter(object):
    def __init__(self, words):
        self.prefix_root = [set(), [None for _ in range(26)]]   
        self.suffix_root = [set(), [None for _ in range(26)]]   
        self.weights = {}                                       
        def insert(word, forwards):                             
            if forwards:
                node = self.prefix_root
                iterate_word = word
            else:
                node = self.suffix_root
                iterate_word = word[::-1]
            node[0].add(word)
            for c in iterate_word:
                if not node[1][ord(c) - ord("a")]:              
                    node[1][ord(c) - ord("a")] = [set(), [None for _ in range(26)]]
                node = node[1][ord(c) - ord("a")]
                node[0].add(word)
        for weight, word in enumerate(words):
            self.weights[word] = weight
            insert(word, True)
            insert(word, False)
    def f(self, prefix, suffix):
        def find_words(word, forwards):
            if forwards:
                node = self.prefix_root
                iterate_word = word
            else:
                node = self.suffix_root
                iterate_word = word[::-1]
            for c in iterate_word:
                node = node[1][ord(c) - ord("a")]
                if not node:
                    return -1       
            return node[0]
        prefix_matches = find_words(prefix, True)
        suffix_matches = find_words(suffix, False)
        if prefix_matches == -1 or suffix_matches == -1:
            return -1
        matches = prefix_matches & suffix_matches
        weight = -1
        for match in matches:
            weight = max(weight, self.weights[match])
        return weight
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def equationsPossible(self, equations):
        graph = [[] for _ in range(26)]         
        not_equal = []                          
        for eqn in equations:
            first, op, second = eqn[0], eqn[1], eqn[3]
            first, second = ord(first) - ord("a"), ord(second) - ord("a")   
            if op == "=":
                graph[first].append(second)
                graph[second].append(first)
            else:
                not_equal.append((first, second))
        groups = [None] * 26                    
        def dfs(node, group):
            if groups[node] is None:
                groups[node] = group
                for nbor in graph[node]:
                    dfs(nbor, group)
        for i in range(26):                     
            dfs(i, i)
        for first, second in not_equal:
            if groups[first] == groups[second]:
                return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def multiply(self, A, B):
        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])      
        C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for r in range(rows_A):
            for c in range(cols_A):
                if A[r][c] != 0:
                    for i in range(cols_B):
                        C[r][i] += A[r][c] * B[c][i]
        return C
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyHashSet(object):
    def __init__(self):
        self.size = 10000
        self.hashset = [[] for _ in range(self.size)]
    def hash_function(self, key):
        return key % self.size          
    def add(self, key):
        if not self.contains(key):      
            self.hashset[self.hash_function(key)].append(key)
    def remove(self, key):
        if self.contains(key):          
            self.hashset[self.hash_function(key)].remove(key)
    def contains(self, key):
        return key in self.hashset[self.hash_function(key)] 
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findRelativeRanks(self, nums):
        num_i = [(num, i) for i, num in enumerate(nums)]
        num_i.sort(reverse=True)
        result = [None for _ in range(len(nums))]
        medals = ["Gold", "Silver", "Bronze"]
        for rank, (_, i) in enumerate(num_i):
            if rank < 3:                                
                result[i] = medals[rank] + " Medal"
            else:
                result[i] = str(rank + 1)               
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        window = set()      
        for i, num in enumerate(nums):
            if i > k:
                window.remove(nums[i - k - 1])
            if num in window:
                return True
            window.add(num)
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def reverseWords(self, s):
        self.reverse(s, 0, len(s)-1)
        s.append(' ')   
        start = 0
        for i in range(len(s)):
            if s[i] == ' ':
                self.reverse(s, start, i-1)     
                start = i+1                     
        s.pop()
    def reverse(self, s, left, right):
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
class Solution2:
    def reverseWords(self, s):
        s.reverse()
        s.append(' ')
        start = 0
        for i in range(len(s)):
            if s[i] == ' ':
                s[start:i] = reversed(s[start:i])
                start = i+1
        s.pop()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findItinerary(self, tickets):
        tickets.sort(reverse = True)    
        flights = defaultdict(list)
        for start, end in tickets:      
            flights[start].append(end)
        journey = []
        def visit(airport):
            while flights[airport]:
                visit(flights[airport].pop())
            journey.append(airport)
        visit("JFK")
        return journey[::-1]
class Solution2(object):
    def findItinerary(self, tickets):
        flights = defaultdict(list)
        tickets.sort(reverse = True)
        for start, end in tickets:
            flights[start].append(end)
        route, stack = [], ['JFK']
        while stack:
            while flights[stack[-1]]:                   
                stack.append(flights[stack[-1]].pop())  
            route.append(stack.pop())                   
        return route[::-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minFlipsMonoIncr(self, S):
        zeros, ones = S.count("0"), 0
        result = zeros
        for c in S:
            if c == "0":
                zeros -= 1
            else:
                ones += 1
            result = min(result, zeros + ones)
        return result
EOF
def camelcase(s):
    res = 1
    for let in s:
        if let.isupper():
            res += 1
    if not s:
        res = 0
    return res
if __name__ == "__main__":
    s = input().strip()
    result = camelcase(s)
    print(result)
EOF
def gridChallenge(grid):
    res = 'YES'
    newgrid = []
    for row in grid:
        newgrid.append(sorted(row))
    for ind in range(len(grid)):
        for jnd in range(ind, len(grid[0])):
            newgrid[ind][jnd], newgrid[jnd][ind] = newgrid[jnd][ind], newgrid[ind][jnd]
    for row in newgrid:
        if row != sorted(row):
            res = 'NO'
            break
    return res
if __name__ == "__main__":
    t = int(input().strip())
    for _ in range(t):
        n = int(input().strip())
        grid = []
        grid_i = 0
        for grid_i in range(n):
            grid_t = str(input().strip())
            grid.append(grid_t)
        result = gridChallenge(grid)
        print(result)
EOF
def minimumAbsoluteDifference(n, arr):
    arr = sorted(arr)
    res = 10**9
    for ind in range(1, len(arr)):
        res = min(res, arr[ind] - arr[ind-1])
    return res
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = minimumAbsoluteDifference(n, arr)
    print(result)
EOF
def migratoryBirds(n, ar):
    count = [0] * 6
    for bird in ar:
        count[bird] += 1
    return count.index(max(count))
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = migratoryBirds(n, ar)
print(result)
EOF
def solve(grades):
    res = []
    for el in grades:
        if el < 38:
            res.append(el)
        elif el%5 >= 3:
            res.append(el + 5-el%5)
        else:
            res.append(el)
    return res
n = int(input().strip())
grades = []
grades_i = 0
for grades_i in range(n):
   grades_t = int(input().strip())
   grades.append(grades_t)
result = solve(grades)
print ("\n".join(map(str, result)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def smallestRange(self, nums):
        n = len(nums)                                       
        window = [(nums[i][0], 0, i) for i in range(n)]     
        heapq.heapify(window)
        heap_min, heap_max = window[0][0], max([nums[i][0] for i in range(n)])
        best_min, best_max = heap_min, heap_max
        while True:
            _, i, i_list = heapq.heappop(window)           
            if i + 1 >= len(nums[i_list]):                  
                return [best_min, best_max]
            heapq.heappush(window, (nums[i_list][i + 1], i + 1, i_list))    
            heap_min = window[0][0]
            heap_max = max(heap_max, nums[i_list][i + 1])
            if heap_max - heap_min < best_max - best_min:   
                best_min, best_max = heap_min, heap_max
EOF
for _ in range(int(input())):
    a, set_a = input(), set(map(int, input().split()))
    b, set_b = input(), set(map(int, input().split()))
    print(set_a.issubset(set_b))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def validTree(self, n, edges):
        class Solution(object):
            def validTree(self, n, edges):
                def dfs(node):                  
                    nbors = adjacency.pop(node, [])
                    for nbor in nbors:
                        dfs(nbor)
                if len(edges) != n - 1:         
                    return False
                adjacency = {i: [] for i in range(n)}   
                for a, b in edges:
                    adjacency[a].append(b)
                    adjacency[b].append(a)
                dfs(0)
                return not adjacency
class Solution2(object):
    def validTree(self, n, edges):
        def find(node):
            if parents[node] == -1:
                return node
            return find(parents[node])
        if len(edges) != n - 1:
            return False
        parents = [-1] * n
        for a, b in edges:
            a_parent = find(a)
            b_parent = find(b)
            if a_parent == b_parent:    
                return False
            parents[a_parent] = b_parent
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def stoneGame(self, piles):
        return True
EOF
class Node:
    def __init__(self,data):
        self.data = data
        self.next = None 
class Solution: 
    def insert(self,head,data):
            p = Node(data)           
            if head==None:
                head=p
            elif head.next==None:
                head.next=p
            else:
                start=head
                while(start.next!=None):
                    start=start.next
                start.next=p
            return head  
    def display(self,head):
        current = head
        while current:
            print(current.data,end=' ')
            current = current.next
    def removeDuplicates(self,head):
        if not head:
            return head
        cur = head.next
        prev = head
        while cur:
            if prev.data == cur.data:
                prev.next = cur.next
                cur = cur.next
            else:
                prev = cur
                cur = cur.next
        return head
mylist= Solution()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class NumMatrix(object):
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        rows, cols = len(matrix), len(matrix[0])
        for r in range(rows):
            for c in range(cols):
                if c != 0:
                    matrix[r][c] += matrix[r][c-1]
                if r != 0:
                    matrix[r][c] += matrix[r-1][c]
                if c != 0 and r != 0:
                    matrix[r][c] -= matrix[r-1][c-1]
        self.matrix = matrix
    def sumRegion(self, row1, col1, row2, col2):
        region = self.matrix[row2][col2]
        if col1 != 0:
            region -= self.matrix[row2][col1-1]
        if row1 != 0:
            region -= self.matrix[row1-1][col2]
        if row1 !=0 and col1 != 0:
            region += self.matrix[row1-1][col1-1]
        return region
EOF
class Node:
    def __init__(self,data):
        self.right=self.left=None
        self.data = data
class Solution:
    def insert(self,root,data):
        if root==None:
            return Node(data)
        else:
            if data<=root.data:
                cur=self.insert(root.left,data)
                root.left=cur
            else:
                cur=self.insert(root.right,data)
                root.right=cur
        return root
    def levelOrder(self,root):
        queue = [root] if root else []
        while queue:
            node = queue.pop()
            print(node.data, end=" ")
            if node.left: queue.insert(0,node.left)
            if node.right: queue.insert(0,node.right)
T=int(input())
myTree=Solution()
root=None
for i in range(T):
    data=int(input())
    root=myTree.insert(root,data)
myTree.levelOrder(root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLHS(self, nums):
        freq = Counter(nums)
        max_harmonious = 0
        for num, count in freq.items():
            if num + 1 in freq:         
                max_harmonious = max(max_harmonious, count + freq[num + 1])
        return max_harmonious
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def __init__(self):         
        self.prev = None
    def flatten(self, root):
        if not root:
            return None
        self.prev = root        
        self.flatten(root.left)
        temp = root.right       
        root.right = root.left
        root.left = None        
        self.prev.right = temp  
        self.flatten(temp)
EOF
def print_full_name(a, b):
    print('Hello {a} {b}! You just delved into python.'.format(a=a, b=b))
print_full_name(input(), input())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numComponents(self, head, G):
        H = set(G)              
        count = 0
        connected = False       
        while head:
            if head.val in H and not connected:     
                connected = True
                count += 1
            elif head.val not in G and connected:   
                connected = False
            head = head.next
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def trapRainWater(self, heightMap):
        if not heightMap or not heightMap[0]:
            return 0
        rows, cols = len(heightMap), len(heightMap[0])
        water = 0
        q = []
        for r in range(rows):
            heapq.heappush(q, (heightMap[r][0], r, 0))
            heapq.heappush(q, (heightMap[r][cols - 1], r, cols - 1))
        for c in range(1, cols - 1):
            heapq.heappush(q, (heightMap[0][c], 0, c))
            heapq.heappush(q, (heightMap[rows - 1][c], rows - 1, c))
        visited = {(r, c) for _, r, c in q}
        while q:
            h, r, c = heapq.heappop(q)
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                r1, c1 = r + dr, c + dc
                if (r1, c1) not in visited and r1 >= 0 and c1 >= 0 and r1 < rows and c1 < cols:
                    visited.add((r1, c1))
                    water += max(0, h - heightMap[r1][c1])
                    heapq.heappush(q, (max(h, heightMap[r1][c1]), r1, c1))
        return water
EOF
def recursive_bubble_sort(arr, steps=None, swapped_count=None):
    if steps is None:
        steps = 0
    if swapped_count is None:
        swapped_count = 0
    swapped = False
    n = len(arr)
    if n is 1:
        return False
    for i in range(n - steps - 1):
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            swapped_count += 1
            swapped = True
    if swapped is True:
        steps += 1
        recursive_bubble_sort(arr, steps, swapped_count)
    else:
        print("Array is sorted in " + str(swapped_count) + " swaps.")
        print('First Element: ' + str(arr[0]))
        print('Last Element: ' + str(arr[n - 1]))
n = int(input().strip())
a = list(map(int, input().strip().split(' ')))
recursive_bubble_sort(a)
EOF
n, m = map(int, input().split())
a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')
EOF
def toys(w):
    w = sorted(w)
    res = 0
    limit = -1
    for el in w:
        if el > limit:
            limit = el + 4
            res += 1
    return res
if __name__ == "__main__":
    n = int(input().strip())
    w = list(map(int, input().strip().split(' ')))
    result = toys(w)
    print(result)
EOF
def makingAnagrams(s1, s2):
    res = 0
    cnt1 = Counter(s1)
    cnt2 = Counter(s2)
    cnt3 = {}
    for let, val in cnt1.items():
        cnt3[let] = abs(val - cnt2[let])
    for let, val in cnt2.items():
        cnt3[let] = abs(val - cnt1[let])
    for el in cnt3.values():
        res += el
    return res
s1 = input().strip()
s2 = input().strip()
result = makingAnagrams(s1, s2)
print(result)
EOF
class Solution(object):
    def maximum69Number (self, num):
        curr, base, change = num, 3, 0
        while curr:
            if curr%10 == 6:
                change = base
            base *= 10
            curr //= 10
        return num+change
class Solution2(object):
    def maximum69Number (self, num):
        return int(str(num).replace('6', '9', 1))
EOF
class Solution(object):
    def isPrefixOfWord(self, sentence, searchWord):
        def KMP(text, pattern):
            def getPrefix(pattern):
                prefix = [-1] * len(pattern)
                j = -1
                for i in xrange(1, len(pattern)):
                    while j > -1 and pattern[j + 1] != pattern[i]:
                        j = prefix[j]
                    if pattern[j + 1] == pattern[i]:
                        j += 1
                    prefix[i] = j
                return prefix
            prefix = getPrefix(pattern)
            j = -1
            for i in xrange(len(text)):
                while j != -1 and pattern[j+1] != text[i]:
                    j = prefix[j]
                if pattern[j+1] == text[i]:
                    j += 1
                if j+1 == len(pattern):
                    return i-j
            return -1
        if sentence.startswith(searchWord):
            return 1
        p = KMP(sentence, ' ' + searchWord)
        if p == -1:
            return -1
        return 1+sum(sentence[i] == ' ' for i in xrange(p+1))
EOF
def is_prime(num):
    if num < 2:
        return False
    else:
        sqrt = int(num**(1/2))
        for i in range(2,sqrt+1):
            if n % i == 0:
                return False
    return True
p = int(input().strip())
for a0 in range(p):
    n = int(input().strip())
    if is_prime(n):
        print("Prime")
    else:
        print("Not prime")
EOF
class Solution(object):
    def findLucky(self, arr):
        count = collections.Counter(arr)
        result = -1
        for k, v in count.iteritems():
            if k == v:
                result = max(result, k)
        return result
EOF
def catAndMouse(x, y, z):
    cat_a = abs(x - z)
    cat_b = abs(y - z)
    if cat_a < cat_b:
        return "Cat A"
    elif cat_a > cat_b:
        return "Cat B"
    else:
        return "Mouse C"
if __name__ == "__main__":
    q = int(input().strip())
    for a0 in range(q):
        x, y, z = input().strip().split(' ')
        x, y, z = [int(x), int(y), int(z)]
        result = catAndMouse(x, y, z)
        print ("".join(map(str, result)))
EOF
def hurdleRace(k, height):
    return max(0, max(height) - k)
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    height = list(map(int, input().strip().split(' ')))
    result = hurdleRace(k, height)
    print(result)
EOF
def staircase(n):
    for i in range(1,n+1):
        print(" "*(n-i)+"
if __name__ == '__main__':
    n = int(input())
    staircase(n)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def buildTree(self, inorder, postorder):
        if not inorder:     
            return None
        inorder_index = inorder.index(postorder.pop())
        root = TreeNode(inorder[inorder_index])
        root.right = self.buildTree(inorder[inorder_index+1:], postorder)   
        root.left = self.buildTree(inorder[:inorder_index], postorder)
        return root
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isValidSerialization(self, preorder):
        if not preorder:
            return True
        expected_leaves = 1
        for node in preorder.split(","):
            if expected_leaves == 0:
                return False
            if node == "
                expected_leaves -= 1
            else:
                expected_leaves += 1
        return expected_leaves == 0
EOF
class UnionFind(object):
    def __init__(self, n):
        self.set = range(n)
        self.count = n
    def find_set(self, x):
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])  
        return self.set[x]
    def union_set(self, x, y):
        x_root, y_root = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[max(x_root, y_root)] = min(x_root, y_root)
        self.count -= 1
        return True
class Solution(object):
    def minCostToSupplyWater(self, n, wells, pipes):
        w = [[c, 0, i] for i, c in enumerate(wells, 1)]
        p = [[c, i, j] for i, j, c in pipes]
        result = 0
        union_find = UnionFind(n+1)
        for c, x, y in sorted(w+p):
            if not union_find.union_set(x, y):
                continue
            result += c
            if union_find.count == 1:
                break
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findKthNumber(self, m, n, k):
        if m > n:                   
            m, n = n, m
        def helper(guess):          
            count = 0
            for i in range(1, m + 1):
                temp = guess // i
                if temp > n:        
                    count += n
                else:
                    count += temp
                if count >= k:      
                    return True
            return False
        left, right = 1, m * n
        while left < right:
            mid = (left + right) // 2
            if helper(mid):
                right = mid
            else:
                left = mid + 1
        return left
EOF
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    first = max(arr)
    print(max(list(filter(lambda a: a != first, arr))))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getMinimumDifference(self, root):
        self.min_diff = float("inf")
        self.prev = float("-inf")
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            self.min_diff = min(self.min_diff, node.val - self.prev)
            self.prev = node.val
            inorder(node.right)
        inorder(root)
        return self.min_diff
EOF
arr = []
for arr_i in range(6):
    arr_t = [int(arr_temp) for arr_temp in input().strip().split(' ')]
    arr.append(arr_t)
best = 0
for ind_i in range(6-2):
    for ind_j in range(6-2):
        temp = 0
        temp += arr[ind_i][ind_j] + arr[ind_i][ind_j+1] + arr[ind_i][ind_j+2]
        temp += arr[ind_i+1][ind_j+1]
        temp += arr[ind_i+2][ind_j] + arr[ind_i+2][ind_j+1] + arr[ind_i+2][ind_j+2]
        if ind_i == 0 and ind_j == 0:
            best = temp
        else:
            best = max(best, temp)
print(best)
EOF
def twoArrays(k, A, B):
    res = 'YES'
    A = sorted(A)
    B = sorted(B, reverse=True)
    for el in zip(A, B):
        if sum(el) < k:
            res = 'NO'
            break
    return res
if __name__ == "__main__":
    q = int(input().strip())
    for a0 in range(q):
        n, k = input().strip().split(' ')
        n, k = [int(n), int(k)]
        A = list(map(int, input().strip().split(' ')))
        B = list(map(int, input().strip().split(' ')))
        result = twoArrays(k, A, B)
        print(result)
EOF
class Solution(object):
    def expand(self, S):  
        def form_words(options):
            words = map("".join, itertools.product(*options))
            words.sort()
            return words
        def generate_option(expr, i):
            option_set = set()
            while i[0] != len(expr) and expr[i[0]] != "}":
                i[0] += 1  
                for option in generate_words(expr, i):
                    option_set.add(option)
            i[0] += 1  
            option = list(option_set)
            option.sort()
            return option
        def generate_words(expr, i):
            options = []
            while i[0] != len(expr) and expr[i[0]] not in ",}":
                tmp = []
                if expr[i[0]] not in "{,}":
                    tmp.append(expr[i[0]])
                    i[0] += 1  
                elif expr[i[0]] == "{":
                    tmp = generate_option(expr, i)
                options.append(tmp)
            return form_words(options)
        return generate_words(S, [0])
class Solution2(object):
    def expand(self, S):  
        def form_words(options):
            words = []
            total = 1
            for opt in options:
                total *= len(opt)
            for i in xrange(total):
                tmp = []
                for opt in reversed(options):
                    i, c = divmod(i, len(opt))
                    tmp.append(opt[c])
                tmp.reverse()
                words.append("".join(tmp))
            words.sort()
            return words
        def generate_option(expr, i):
            option_set = set()
            while i[0] != len(expr) and expr[i[0]] != "}":
                i[0] += 1  
                for option in generate_words(expr, i):
                    option_set.add(option)
            i[0] += 1  
            option = list(option_set)
            option.sort()
            return option
        def generate_words(expr, i):
            options = []
            while i[0] != len(expr) and expr[i[0]] not in ",}":
                tmp = []
                if expr[i[0]] not in "{,}":
                    tmp.append(expr[i[0]])
                    i[0] += 1  
                elif expr[i[0]] == "{":
                    tmp = generate_option(expr, i)
                options.append(tmp)
            return form_words(options)
        return generate_words(S, [0])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def cutOffTree(self, forest):
        rows, cols = len(forest), len(forest[0])
        trees = sorted((h, r, c) for r, row in enumerate(forest)    
                       for c, h in enumerate(row) if h > 1)
        to_visit = set((r, c) for _, r, c in trees)                 
        visited = set()
        queue = [(0, 0)]
        while queue:
            r, c = queue.pop()
            to_visit.discard((r, c))
            visited.add((r, c))
            for r1, c1 in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
                if 0 <= r1 < rows and 0 <= c1 < cols and forest[r1][c1] and (r1, c1) not in visited:
                    queue.append((r1, c1))
        if to_visit:            
            return -1
        def distance(r1, c1, r2, c2):
            direct = abs(r1 - r2) + abs(c1 - c2)  
            diversions = 0
            queue = [(r1, c1)]  
            next_queue = []     
            visited = set()
            while True:
                if not queue:  
                    queue, next_queue = next_queue, []  
                    diversions += 1
                r1, c1 = queue.pop()
                if (r1, c1) == (r2, c2):  
                    return direct + diversions * 2  
                if (r1, c1) in visited:
                    continue
                visited.add((r1, c1))
                for r1, c1, closer in (r1 + 1, c1, r1 < r2), (r1 - 1, c1, r1 > r2), (r1, c1 + 1, c1 < c2), (
                r1, c1 - 1, c1 > c2):
                    if 0 <= r1 < rows and 0 <= c1 < cols and forest[r1][c1]:
                        (queue if closer else next_queue).append((r1, c1))
        result = 0
        r1, c1 = 0, 0                               
        for _, r2, c2 in trees:
            result += distance(r1, c1, r2, c2)      
            r1, c1 = r2, c2                         
        return result
EOF
class MyQueue(object):
    def __init__(self):
        self.q = []
    def peek(self):
        return self.q[0]
    def pop(self):
        self.q.pop(0)
    def put(self, value):
        self.q.append(value)
queue = MyQueue()
t = int(input())
for line in range(t):
    values = map(int, input().split())
    values = list(values)
    if values[0] == 1:
        queue.put(values[1])        
    elif values[0] == 2:
        queue.pop()
    else:
        print(queue.peek())
EOF
class Solution(object):
    def sumEvenAfterQueries(self, A, queries):
        total = sum(v for v in A if v % 2 == 0)
        result = []
        for v, i in queries:
            if A[i] % 2 == 0:
                total -= A[i]
            A[i] += v
            if A[i] % 2 == 0:
                total += A[i]
            result.append(total)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class StringIterator(object):
    def __init__(self, compressedString):
        self.letter = None
        self.count = 0              
        self.i = 0                  
        self.s = compressedString
    def next(self):
        if not self.hasNext():
            return " "
        if self.count == 0:
            self.move()
        self.count -= 1             
        return self.letter
    def hasNext(self):
        return self.count > 0 or self.i < len(self.s)
    def move(self):
        self.letter = self.s[self.i]
        self.count = 0
        self.i += 1
        while self.i < len(self.s) and self.s[self.i] <= "9":   
            self.count = self.count * 10 + int(self.s[self.i])  
            self.i += 1
EOF
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def sumRootToLeaf(self, root):
        M = 10**9 + 7
        def sumRootToLeafHelper(root, val):
            if not root:
                return 0
            val = (val*2 + root.val) % M
            if not root.left and not root.right:
                return val
            return (sumRootToLeafHelper(root.left, val) +
                    sumRootToLeafHelper(root.right, val)) % M
        return sumRootToLeafHelper(root, 0)
EOF
def chessboardGame(x, y):
    x = x%4
    y = y%4
    if x in [1, 2] and y in [1,2]:
        return 'Second'
    else:
        return 'First'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        xy = input().split()
        x = int(xy[0])
        y = int(xy[1])
        result = chessboardGame(x, y)
        fptr.write(result + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def levelOrder(self, root):
        result = []
        if not root:
            return result
        level_nodes = [root]
        while level_nodes:
            new_level_nodes = []
            result.append([])
            for node in level_nodes:
                result[-1].append(node.val)
                if node.left:
                    new_level_nodes.append(node.left)
                if node.right:
                    new_level_nodes.append(node.right)
            level_nodes = new_level_nodes
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def wordSubsets(self, A, B):
        required = defaultdict(int)  
        for b in B:
            freq = Counter(b)
            for letter, count in freq.items():
                required[letter] = max(required[letter], count)
        results = []
        for a in A:
            freq = Counter(a)
            if all(freq[letter] >= count for letter, count in required.items()):
                results.append(a)
        return results
EOF
def product(fracs):
    t = reduce(lambda x, y : x * y, fracs)
    return t.numerator, t.denominator
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def sortedListToBST(self, head):
        count = 0
        node = head
        while node:
            count += 1
            node = node.next
        return self.list_to_bst([head], 0, count - 1)   
    def list_to_bst(self, node_as_list, start, end):  
        if start > end:
            return None
        mid = (start + end) // 2
        left_subtree = self.list_to_bst(node_as_list, start, mid - 1)   
        root = TreeNode(node_as_list[0].val)
        root.left = left_subtree
        node_as_list[0] = node_as_list[0].next      
        root.right = self.list_to_bst(node_as_list, mid + 1, end)
        return root
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rotateString(self, A, B):
        if len(A) != len(B):
            return False
        return A in B + B
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, rects):
        self.cumul_area = [0]       
        self.x_dimensions = [0]     
        self.rects = rects
        for x1, y1, x2, y2 in rects:
            x_dim, y_dim = x2 - x1 + 1, y2 - y1 + 1
            self.x_dimensions.append(x_dim)
            self.cumul_area.append(self.cumul_area[-1] + x_dim * y_dim)
    def pick(self):
        n = random.randint(1, self.cumul_area[-1])      
        i = bisect.bisect_left(self.cumul_area, n)      
        n -= (self.cumul_area[i - 1] + 1)               
        dy, dx = divmod(n, self.x_dimensions[i])        
        x, y = self.rects[i - 1][:2]
        return [x + dx, y + dy]
EOF
def kangaroo(x1, v1, x2, v2):
    if x1 >= x2 and v1 > v2:
        return 'NO'
    elif x2 >= x1 and v2 > v1:
        return 'NO'
    elif (x1 > x2 or x2 > x1) and v1 == v2:
        return 'NO'
    else:
        pos1, pos2 = x1, x2
        dif_prev = dif = abs(pos1 - pos2)
        while dif_prev >= dif:
            if pos1 == pos2:
                return 'YES'
            pos1 += v1
            pos2 += v2
            dif_prev, dif = dif, abs(pos1 - pos2)
    return 'NO'
x1, v1, x2, v2 = input().strip().split(' ')
x1, v1, x2, v2 = [int(x1), int(v1), int(x2), int(v2)]
result = kangaroo(x1, v1, x2, v2)
print(result)
EOF
def simpleArraySum(ar):
    summ = 0
    for i in ar:
        summ += i
    return summ 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = simpleArraySum(ar)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def compare_lists(llist1, llist2):
    if not llist1 and not llist2:
        return True
    if not llist1 or not llist2:
        return False
    if llist1.data != llist2.data:
        return False
    return compare_lists(llist1.next,llist2.next)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        llist1_count = int(input())
        llist1 = SinglyLinkedList()
        for _ in range(llist1_count):
            llist1_item = int(input())
            llist1.insert_node(llist1_item)
        llist2_count = int(input())
        llist2 = SinglyLinkedList()
        for _ in range(llist2_count):
            llist2_item = int(input())
            llist2.insert_node(llist2_item)
        result = compare_lists(llist1.head, llist2.head)
        fptr.write(str(int(result)) + '\n')
    fptr.close()
EOF
def nimbleGame(s):
    if len(s) > 1:
        res = 0
        for ind, el in enumerate(s):
            if el%2 == 1:
                res ^= ind
    else:
        return 'Second'
    if res == 0:
        return 'Second'
    else:
        return 'First'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        s = list(map(int, input().rstrip().split()))
        result = nimbleGame(s)
        fptr.write(result + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ExamRoom(object):
    def __init__(self, N):
        self.seats = []
        self.N = N
    def seat(self):
        if not self.seats:                      
            self.seats.append(0)
            return 0
        max_dist, index = self.seats[0], 0      
        for i in range(len(self.seats) - 1):    
            dist = (self.seats[i + 1] - self.seats[i]) // 2     
            if dist > max_dist:                 
                max_dist = dist
                index = self.seats[i] + dist
        if self.N - 1 - self.seats[-1] > max_dist:  
            index = self.N - 1
        bisect.insort(self.seats, index)        
        return index
    def leave(self, p):
        self.seats.remove(p)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minTotalDistance(self, grid):
        rows, cols = [], []
        for r in range(len(grid)):      
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    rows.append(r)
                    cols.append(c)
        cols.sort()     
        dist = 0
        left, right = 0, len(rows)-1    
        while left < right:
            dist += (rows[right] - rows[left])
            left += 1
            right -= 1
        left, right = 0, len(cols)-1    
        while left < right:
            dist += (cols[right] - cols[left])
            left += 1
            right -= 1
        return dist
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def numMusicPlaylists(self, N, L, K):
        used_count = {0: 1}     
        for song in range(L):
            new_used_count = defaultdict(int)
            for used, count in used_count.items():
                new_used_count[used + 1] += count * (N - used)  
                if used > K:
                    new_used_count[used] += count * (used - K)  
            used_count = new_used_count
        return used_count[N] % (10 ** 9 + 7)
EOF
if __name__ == "__main__":
    string = input()
    sub = input()
    matches = list(re.finditer(r'(?={})'.format(sub), string))
    if matches:
        for match in matches:
            print((match.start(), match.end() + len(sub) - 1))
    else:
        print((-1, -1))
EOF
def checkNode(node, min, max):
    if node == None:
        return True
    if node.data <= min or node.data >= max:
        return False
    return checkNode(node.left, min, node.data) and checkNode(node.right, node.data, max)
def checkBST(root):
    return checkNode(root, float('-inf'), float('inf'))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProfitAssignment(self, difficulty, profit, worker):
        max_profits = list(zip(difficulty, profit))
        max_profits.sort()
        max_profits.append((float("inf"), 0))
        total_profit = 0
        best_profit = 0
        i = 0
        worker.sort()
        for diff in worker:
            while max_profits[i][0] <= diff:                        
                best_profit = max(best_profit, max_profits[i][1])   
                i += 1
            total_profit += best_profit
        return total_profit
EOF
def minimumSwaps(arr):
    res = 0
    ind = 0
    while ind < len(arr)-1:
        if arr[ind] == ind + 1:
            ind += 1
            continue
        else:
            arr[arr[ind]-1], arr[ind] = arr[ind], arr[arr[ind]-1]
            res += 1
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    res = minimumSwaps(arr)
    fptr.write(str(res) + '\n')
    fptr.close()
EOF
class Solution(object):
    def findItinerary(self, tickets):
        def route_helper(origin, ticket_cnt, graph, ans):
            if ticket_cnt == 0:
                return True
            for i, (dest, valid)  in enumerate(graph[origin]):
                if valid:
                    graph[origin][i][1] = False
                    ans.append(dest)
                    if route_helper(dest, ticket_cnt - 1, graph, ans):
                        return ans
                    ans.pop()
                    graph[origin][i][1] = True
            return False
        graph = collections.defaultdict(list)
        for ticket in tickets:
            graph[ticket[0]].append([ticket[1], True])
        for k in graph.keys():
            graph[k].sort()
        origin = "JFK"
        ans = [origin]
        route_helper(origin, len(tickets), graph, ans)
        return ans
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rectangleArea(self, rectangles):
        x_events = []                       
        for i, (x1, y1, x2, y2) in enumerate(rectangles):
            x_events.append((x1, True, i))
            x_events.append((x2, False, i))
        x_events.sort()
        area = 0
        alive = set()                       
        y_coverage = 0                      
        x_prev = 0
        for x, start, i in x_events:
            area += (x - x_prev) * y_coverage   
            x_prev = x
            if start:                       
                alive.add(i)
            else:
                alive.discard(i)
            y_events = []                   
            for i in alive:
                y_events.append((rectangles[i][1], 1))  
                y_events.append((rectangles[i][3], -1)) 
            y_events.sort()
            y_coverage = 0
            prev_y = 0
            alive_y = 0                     
            for y, start_y in y_events:
                if alive_y > 0:             
                    y_coverage += y - prev_y
                alive_y += start_y          
                prev_y = y
        return area % (10 ** 9 + 7)
EOF
class Solution(object):
    def uniqueLetterString(self, S):
        M = 10**9 + 7
        index = {c: [-1, -1] for c in string.ascii_uppercase}
        result = 0
        for i, c in enumerate(S):
            k, j = index[c]
            result += (i-j) * (j-k)
            index[c] = [j, i]
        for c in index:
            k, j = index[c]
            result += (len(S)-j) * (j-k)
        return result % M
EOF
def SortedInsert(head, data):
    if head == None:
        return Node(data=data, next_node=None, prev_node=None)
    node = head
    while node.next != None and node.next.data <= data:
        node = node.next
    node.next = Node(data=data, next_node=node.next, prev_node=node)
    return head
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def addBoldTag(self, s, dict):
        in_tag = [False for _ in range(len(s))]  
        start_letters = defaultdict(list)       
        for word in dict:
            start_letters[word[0]].append(word)
        matches = []                            
        for i, c in enumerate(s):
            new_matches = []
            for word, word_index in matches:
                if c == word[word_index + 1]:
                    if word_index + 1 == len(word) - 1:     
                        for j in range(i - len(word) + 1, i + 1):
                            in_tag[j] = True
                    else:
                        new_matches.append([word, word_index + 1])  
            for word in start_letters[c]:                   
                if len(word) == 1:
                    in_tag[i] = True
                else:
                    new_matches.append([word, 0])
            matches = new_matches
        result = []
        for i, c in enumerate(s):
            if in_tag[i] and (i == 0 or not in_tag[i - 1]): 
                result.append("<b>")
            elif not in_tag[i] and (i != 0 and in_tag[i - 1]):  
                result.append("</b>")
            result.append(c)
        if in_tag[-1]:
            result.append("</b>")   
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def exclusiveTime(self, n, logs):
        stack = []
        exclusive = [0 for _ in range(n)]
        start = None
        for log in logs:
            fn, state, time = log.split(":")
            fn, time = int(fn), int(time)
            if state == "start":
                if stack:
                    exclusive[stack[-1]] += time - start
                stack.append(fn)
                start = time
            else:
                exclusive[stack.pop()] += time - start + 1
                start = time + 1
        return exclusive
EOF
def marsExploration(s):
    res = 0
    for ind in range(0, len(s), 3):
        if s[ind] != 'S':
            res += 1
        if s[ind+1] != 'O':
            res += 1
        if s[ind+2] != 'S':
            res += 1
    return res
if __name__ == "__main__":
    s = input().strip()
    result = marsExploration(s)
    print(result)
EOF
allres = []
def k_factorization(n, arr, cur, result):
    if cur == n:
        allres.append(list(result))
        return True
    elif cur > n:
        return False
    for ind, el in enumerate(arr):
        result.append(cur*el)
        k_factorization(n, arr[ind:], cur*el, result)
        result.pop()
    return False
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    arr = sorted(list(map(int, input().strip().split(' '))))
    result = [1]
    k_factorization(n, arr, 1, result)
    if not allres:
        print(-1)
    else:
        print(" ".join(list(map(str, min(allres, key = len)))))
EOF
class Solution(object):
    def widthOfBinaryTree(self, root):
        def dfs(node, i, depth, leftmosts):
            if not node:
                return 0
            if depth >= len(leftmosts):
                leftmosts.append(i)
            return max(i-leftmosts[depth]+1, \
                       dfs(node.left, i*2, depth+1, leftmosts), \
                       dfs(node.right, i*2+1, depth+1, leftmosts))
        leftmosts = []
        return dfs(root, 1, 0, leftmosts)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def rightSideView(self, root):
        if not root:
            return []
        right_view = []
        layer = [root]
        while layer:
            right_view.append(layer[-1].val)
            next_layer = []
            for node in layer:
                if node.left:
                    next_layer.append(node.left)
                if node.right:
                    next_layer.append(node.right)
            layer = next_layer
        return right_view
class Solution2(object):
    def rightSideView(self, root):
        right_side = []
        self.recursive(root, 0, right_side)
        return right_side
    def recursive(self, node, depth, right_side):
        if not node:
            return
        if depth >= len(right_side):        
            right_side.append(node.val)
        else:
            right_side[depth] = node.val
        self.recursive(node.left, depth+1, right_side)  
        self.recursive(node.right, depth+1, right_side)
EOF
if __name__ == "__main__":
    arr1 = list(map(int, input().strip().split(' ')))
    arr2 = list(map(int, input().strip().split(' ')))
    for el in product(arr1, arr2):
        print("{} ".format(el), end='')
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def largestNumber(self, nums):
        def comparator(x, y):   
            if x+y > y+x:       
                return -1       
            else:
                return 1
        nums = list(map(str, nums))     
        nums.sort(key=functools.cmp_to_key(comparator))
        return str(int("".join(nums)))  
EOF
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    head = None
    def sortedListToBST(self, head):
        current, length = head, 0
        while current is not None:
            current, length = current.next, length + 1
        self.head = head
        return self.sortedListToBSTRecu(0, length)
    def sortedListToBSTRecu(self, start, end):
        if start == end:
            return None
        mid = start + (end - start) / 2
        left = self.sortedListToBSTRecu(start, mid)
        current = TreeNode(self.head.val)
        current.left = left
        self.head = self.head.next
        current.right = self.sortedListToBSTRecu(mid + 1, end)
        return current
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canVisitAllRooms(self, rooms):
        visited = set()
        def dfs(room):
            if room in visited:
                return
            visited.add(room)
            for key in rooms[room]:
                dfs(key)
        dfs(0)
        return len(visited) == len(rooms)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def mergeKLists(self, lists):
        prev = dummy = ListNode(None)
        next_nodes, heap = [], []
        for i, node in enumerate(lists):
            next_nodes.append(node)         
            if node:
                heap.append((node.val, i))
        heapq.heapify(heap)
        while heap:
            value, i = heapq.heappop(heap)
            node = next_nodes[i]
            prev.next = node                
            prev = prev.next
            if node.next:
                next_nodes[i] = node.next
                heapq.heappush(heap, (node.next.val, i))
        return dummy.next
EOF
class DoublyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
        self.prev = None
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = DoublyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
            node.prev = self.tail
        self.tail = node
def print_doubly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def reverse(head):
    if not head:
        return None
    previous = None
    while head:
        head.prev = head.next
        head.next = previous 
        previous = head
        head = head.prev
    return previous
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        llist_count = int(input())
        llist = DoublyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        llist1 = reverse(llist.head)
        print_doubly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()
EOF
def solution():
    res = 0
    cnt = 0
    for ind in range(len(topic)-1):
        for jnd in range(ind+1, len(topic)):
            tmp = bin(int(topic[ind], 2) | (int(topic[jnd], 2))).count("1")
            if tmp > res:
                res = tmp
                cnt = 1
            elif tmp == res:
                cnt += 1
    return (res, cnt)
n,m = input().strip().split(' ')
n,m = [int(n),int(m)]
topic = []
topic_i = 0
for topic_i in range(n):
    topic_t = str(input().strip())
    topic.append(topic_t)
print("\n".join(map(str, solution())))
EOF
def divisibleSumPairs(n, k, ar):
    res = 0
    comb = list(combinations(ar, 2))
    for pair in comb:
        if sum(pair) % k == 0:
            res += 1
    return res
n, k = input().strip().split(' ')
n, k = [int(n), int(k)]
ar = list(map(int, input().strip().split(' ')))
result = divisibleSumPairs(n, k, ar)
print(result)
EOF
class Solution(object):
    def findIntegers(self, num):
        dp = [0] * 32
        dp[0], dp[1] = 1, 2
        for i in xrange(2, len(dp)):
            dp[i] = dp[i-1] + dp[i-2]
        result, prev_bit = 0, 0
        for i in reversed(xrange(31)):
            if (num & (1 << i)) != 0:
                result += dp[i]
                if prev_bit == 1:
                    result -= 1
                    break
                prev_bit = 1
            else:
                prev_bit = 0
        return result + 1
EOF
def staircase(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 4
    return staircase(n-1) + staircase(n-2) + staircase(n-3)
def staircase_lin(n):
    st1, st2, st3 = 1, 2, 4
    for _ in range(n-1):
        st1, st2, st3 = st2, st3, st1 + st2 + st3
    return st1
s = int(input().strip())
for a0 in range(s):
    n = int(input().strip())
    print(staircase_lin(n))
EOF
class TrieNode(object):
    def __init__(self):
        self.__TOP_COUNT = 3
        self.infos = []
        self.leaves = {}
    def insert(self, s, times):
        cur = self
        cur.add_info(s, times)
        for c in s:
            if c not in cur.leaves:
                cur.leaves[c] = TrieNode()
            cur = cur.leaves[c]
            cur.add_info(s, times)
    def add_info(self, s, times):
        for p in self.infos:
            if p[1] == s:
                p[0] = -times
                break
        else:
            self.infos.append([-times, s])
        self.infos.sort()
        if len(self.infos) > self.__TOP_COUNT:
            self.infos.pop()
class AutocompleteSystem(object):
    def __init__(self, sentences, times):
        self.__trie = TrieNode()
        self.__cur_node = self.__trie
        self.__search = []
        self.__sentence_to_count = collections.defaultdict(int)
        for sentence, count in zip(sentences, times):
            self.__sentence_to_count[sentence] = count
            self.__trie.insert(sentence, count)
    def input(self, c):
        result = []
        if c == '
            self.__sentence_to_count["".join(self.__search)] += 1
            self.__trie.insert("".join(self.__search), self.__sentence_to_count["".join(self.__search)])
            self.__cur_node = self.__trie
            self.__search = []
        else:
            self.__search.append(c)
            if self.__cur_node:
                if c not in self.__cur_node.leaves:
                    self.__cur_node = None
                    return []
                self.__cur_node = self.__cur_node.leaves[c]
                result = [p[1] for p in self.__cur_node.infos]
        return result
EOF
class Solution(object):
    def subdomainVisits(self, cpdomains):
        result = collections.defaultdict(int)
        for domain in cpdomains:
            count, domain = domain.split()
            count = int(count)
            frags = domain.split('.')
            curr = []
            for i in reversed(xrange(len(frags))):
                curr.append(frags[i])
                result[".".join(reversed(curr))] += count
        return ["{} {}".format(count, domain) \
                for domain, count in result.iteritems()]
EOF
def squares(a, b):
    res = 0
    res = floor(b**0.5)+1 - ceil(a**0.5)
    return res
if __name__ == "__main__":
    q = int(input().strip())
    for a0 in range(q):
        a, b = input().strip().split(' ')
        a, b = [int(a), int(b)]
        result = squares(a, b)
        print(result)
EOF
def super_reduced_string(s):
    if len(s) == 1:
        return s
    ind = 1
    while ind < len(s):
        if s[ind] == s[ind-1]:
            if len(s) == 2:
                return 'Empty String'
            s = s[:ind-1] + s[ind+1:]
            ind = 1
        else:
            ind += 1
    if len(s) == 0:
        return 'Empty String'
    else:
        return s
s = input().strip()
result = super_reduced_string(s)
print(result)
EOF
class OrderedCounter(Counter, OrderedDict):
    pass
if __name__ == '__main__':
    s = input()
    for i in OrderedCounter(sorted(s)).most_common(3):
        print(*i)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def fairCandySwap(self, A, B):
        A_candy, B_candy = sum(A), sum(B)
        difference = (A_candy - B_candy) // 2   
        B_set = set(B)
        for a in A:
            if a - difference in B_set:
                return [a, a - difference]
        return []
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findContestMatch(self, n):
        result = [str(i) for i in range(1, n + 1)]
        while len(result) > 1:
            new_result = []
            for i in range(len(result) // 2):
                new_result.append("(" + result[i] + "," + result[len(result) - i - 1] + ")")
            result = new_result
        return result[0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numDistinctIslands2(self, grid):
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        def BFS(base_r, base_c):
            grid[base_r][base_c] = 0    
            queue = [(base_r, base_c)]
            for r, c in queue:          
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols and grid[new_r][new_c] == 1:
                        grid[new_r][new_c] = 0
                        queue.append((new_r, new_c))
            canonical = []              
            for _ in range(4):          
                queue = [(c, -r) for r, c in queue]     
                min_r, min_c = min([r for r, _ in queue]), min([c for _, c in queue])
                canonical = max(canonical, sorted([(r - min_r, c - min_c) for r, c in queue]))
                reflected = [(r, -c) for r, c in queue] 
                min_r, min_c = min([r for r, _ in reflected]), min([c for _, c in reflected])
                canonical = max(canonical, sorted([(r - min_r, c - min_c) for r, c in reflected]))
            return tuple(canonical)     
        islands = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    continue
                islands.add(BFS(r, c))
        return len(islands)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def singleNumber(self, nums):
        xor = 0
        for num in nums:
            xor ^= num
        return xor
EOF
S = input().strip()
try:
    print(int(S))
except:
    print("Bad String")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reverseString(self, s):
        return s[::-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Block(object):
    def __init__(self, val=0):
        self.val = val
        self.keys = set()
        self.before = None
        self.after = None
    def remove(self):
        self.before.after = self.after
        self.after.before = self.before
        self.before, self.after = None, None
    def insert_after(self, new_block):
        old_after = self.after
        self.after = new_block
        new_block.before = self
        new_block.after = old_after
        old_after.before = new_block
class AllOne(object):
    def __init__(self):
        self.begin = Block()  
        self.end = Block()  
        self.begin.after = self.end
        self.end.before = self.begin
        self.mapping = {}  
    def inc(self, key):
        if not key in self.mapping:  
            current_block = self.begin
        else:
            current_block = self.mapping[key]
            current_block.keys.remove(key)
        if current_block.val + 1 != current_block.after.val:  
            new_block = Block(current_block.val + 1)
            current_block.insert_after(new_block)
        else:
            new_block = current_block.after
        new_block.keys.add(key)  
        self.mapping[key] = new_block  
        if not current_block.keys and current_block.val != 0:  
            current_block.remove()
    def dec(self, key):
        if not key in self.mapping:
            return
        current_block = self.mapping[key]
        del self.mapping[key]  
        current_block.keys.remove(key)
        if current_block.val != 1:
            if current_block.val - 1 != current_block.before.val:  
                new_block = Block(current_block.val - 1)
                current_block.before.insert_after(new_block)
            else:
                new_block = current_block.before
            new_block.keys.add(key)
            self.mapping[key] = new_block
        if not current_block.keys:  
            current_block.remove()
    def getMaxKey(self):
        if self.end.before.val == 0:
            return ""
        key = self.end.before.keys.pop()  
        self.end.before.keys.add(key)
        return key
    def getMinKey(self):
        if self.begin.after.val == 0:
            return ""
        key = self.begin.after.keys.pop()
        self.begin.after.keys.add(key)
        return key
EOF
if __name__ == '__main__':
    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))
    maxx = -81 
    for i in range(4):
        for j in range(4):
            total = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2];
            if total > maxx:
                maxx = total
    print(maxx)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findNthDigit(self, n):
        length = 1
        digits = 9
        while n > digits:
            n -= digits
            digits = (length + 1) * 9 * (10 ** length)
            length += 1
        start = 10 ** (length - 1)          
        num, digit = divmod(n - 1, length)
        num += start
        return int(str(num)[digit])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        if n == 1:
            return [0]
        connections = defaultdict(set)      
        for a, b in edges:
            connections[a].add(b)
            connections[b].add(a)
        leaves = set(node for node in connections if len(connections[node]) == 1)
        while len(connections) > 2:
            new_leaves = set()
            for leaf in leaves:
                nbor = connections[leaf].pop()
                connections[nbor].remove(leaf)
                if len(connections[nbor]) == 1:
                    new_leaves.add(nbor)
                del connections[leaf]
            leaves = new_leaves
        return list(connections.keys())
EOF
def super_reduced_string(s):
    str_list = list(s)
    i = 0
    while i < len((str_list))-1:
        if str_list[i] == str_list[i+1]:
                del str_list[i]
                del str_list[i]
                i = 0
        else:
            i += 1
    if len(str_list) != 0:
        return ''.join(str_list)
    else:
        return 'Empty String'
s = input().strip()
result = super_reduced_string(s)
print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def invertTree(self, root):
        if not root:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
EOF
N, M = map(int, input().split())
arr = [input().strip().split() for _ in range(N)]
arr = np.array(arr, int)
print(np.transpose(arr))
print(arr.flatten())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProduct(self, nums):
        largest_product = float('-inf')
        most_neg, most_pos = 1, 1
        for num in nums:
            most_pos, most_neg = max(num, most_pos * num, most_neg * num), min(num, most_pos * num, most_neg * num)
            largest_product = max(largest_product, most_pos, most_neg)
        return largest_product
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def possibleBipartition(self, N, dislikes):
        dislike = defaultdict(set)          
        for a, b in dislikes:
            dislike[a].add(b)
            dislike[b].add(a)
        this, other = set(), set()          
        for i in range(1, N + 1):
            if i in this or i in other:     
                continue
            to_add = {i}
            while to_add:
                this |= to_add              
                disliked = set()            
                for num in to_add:
                    disliked |= dislike[num]
                if disliked & this:         
                    return False
                disliked -= other           
                to_add = disliked
                this, other = other, this
        return True
EOF
n = int(input().strip())
phonebook = {}
for i in range(n):
    details = [temp_string for temp_string in input().split(' ')]
    phonebook[details[0].lower()] = details[1]
input_flag = True
while(input_flag):
    key = input().strip().lower()
    if key in phonebook:
        print(key + "=" + phonebook[key])
    else:
        print("Not found")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def plusOne(self, digits):
        i = len(digits)-1
        while i >= 0 and digits[i] == 9:
            digits[i] = 0
            i -= 1
        if i == -1:
            return [1] + digits
        return digits[:i] + [digits[i]+1] + digits[i+1:]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findAllConcatenatedWordsInADict(self, words):
        def is_concat(word):
            if not word or word in word_set:    
                return True
            for i in range(1, len(word) + 1):  
                if word[:i] in word_set and is_concat(word[i:]):
                    return True
            return False
        word_set = set(words)
        results = []
        for word in words:
            for i in range(1, len(word)):  
                if word[:i] in word_set and is_concat(word[i:]):
                    results.append(word)
                    break
        return results
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxDepth(self, root):
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findFrequentTreeSum(self, root):
        def count_sums(node):
            if not node:
                return 0
            total_sum = node.val + count_sums(node.left) + count_sums(node.right)
            tree_sums[total_sum] += 1
            return total_sum
        if not root:
            return []
        tree_sums = defaultdict(int)
        count_sums(root)
        max_sum = max(tree_sums.values())
        result = []
        for key, val in tree_sums.items():
            if val == max_sum:
                result.append(key)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        i, j, k = m - 1, n - 1, m + n - 1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        if i < 0:       
            nums1[:k+1] = nums2[:j+1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def networkDelayTime(self, times, N, K):
        best_times = [float("inf") for _ in range(N + 1)]   
        best_times[K] = 0
        network = [[] for _ in range(N + 1)]                
        for u, v, w in times:
            network[u].append((v, w))
        nodes = {n for n in range(1, N + 1)}                
        while nodes:
            best_time = float("inf")
            for node in nodes:                              
                if best_times[node] < best_time:
                    best_time = best_times[node]
                    next_node = node
            if best_time == float("inf"):                   
                return -1
            nodes.remove(next_node)                         
            for nbor, time in network[next_node]:           
                best_times[nbor] = min(best_times[nbor], best_time + time)
        return max(best_times[1:])                          
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def reverseOnlyLetters(self, S):
        letters = set(string.ascii_lowercase + string.ascii_uppercase)
        S = [c for c in S]
        left, right = 0, len(S) - 1
        while left < right:
            while left < right and S[left] not in letters:
                left += 1
            while left < right and S[right] not in letters:
                right -= 1
            S[left], S[right] = S[right], S[left]
            left += 1
            right -= 1
        return "".join(S)
EOF
class Solution:
    def __init__(self):
        self.stack = []
        self.queue = []
    def pushCharacter(self,x):
        self.stack.append(x)
    def enqueueCharacter(self,x):
        self.queue.append(x)
    def popCharacter(self):
        return self.stack.pop()
    def dequeueCharacter(self):
        return self.queue.pop(0)
s=input()
obj=Solution()   
l=len(s)
for i in range(l):
    obj.pushCharacter(s[i])
    obj.enqueueCharacter(s[i])
isPalindrome=True
for i in range(l // 2):
    if obj.popCharacter()!=obj.dequeueCharacter():
        isPalindrome=False
        break
if isPalindrome:
    print("The word, "+s+", is a palindrome.")
else:
    print("The word, "+s+", is not a palindrome.")
EOF
def average(array):
    distinct_height = set(array)
    return sum(distinct_height) / len(distinct_height)
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
EOF
if __name__ == '__main__':
    print("Hello, World!")
EOF
def minimum_index(seq):
    if len(seq) == 0:
        raise ValueError("Cannot get the minimum value index from an empty sequence")
    min_idx = 0
    for i in range(1, len(seq)):
        if seq[i] < seq[min_idx]:
            min_idx = i
    return min_idx
class TestDataEmptyArray(object):
        def get_array():
        return []
class TestDataUniqueValues(object):
        def get_array():
        return [2,1,3]
        def get_expected_result():
        return 1
class TestDataExactlyTwoDifferentMinimums(object):
        def get_array():
        return [2,1,1]
        def get_expected_result():
        return 1
def TestWithEmptyArray():
    try:
        seq = TestDataEmptyArray.get_array()
        result = minimum_index(seq)
    except ValueError as e:
        pass
    else:
        assert False
def TestWithUniqueValues():
    seq = TestDataUniqueValues.get_array()
    assert len(seq) >= 2
    assert len(list(set(seq))) == len(seq)
    expected_result = TestDataUniqueValues.get_expected_result()
    result = minimum_index(seq)
    assert result == expected_result
def TestiWithExactyTwoDifferentMinimums():
    seq = TestDataExactlyTwoDifferentMinimums.get_array()
    assert len(seq) >= 2
    tmp = sorted(seq)
    assert tmp[0] == tmp[1] and (len(tmp) == 2 or tmp[1] < tmp[2])
    expected_result = TestDataExactlyTwoDifferentMinimums.get_expected_result()
    result = minimum_index(seq)
    assert result == expected_result
TestWithEmptyArray()
TestWithUniqueValues()
TestiWithExactyTwoDifferentMinimums()
print("OK")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkInclusion(self, s1, s2):
        n1 = len(s1)
        freq = [0] * 26     
        for c in s1:
            freq[ord(c) - ord("a")] += 1
        for i, c in enumerate(s2):
            freq[ord(c) - ord("a")] -= 1    
            if i >= n1:
                freq[ord(s2[i - n1]) - ord("a")] += 1   
            if not any(freq):
                return True
        return False
EOF
class Solution(object):
    def numEnclaves(self, A):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        def dfs(A, i, j):
            if not (0 <= i < len(A) and 0 <= j < len(A[0]) and A[i][j]):
                return
            A[i][j] = 0
            for d in directions:
                dfs(A, i+d[0], j+d[1])
        for i in xrange(len(A)):
            dfs(A, i, 0)
            dfs(A, i, len(A[0])-1)
        for j in xrange(1, len(A[0])-1):
            dfs(A, 0, j)
            dfs(A, len(A)-1, j)
        return sum(sum(row) for row in A)
EOF
def isBalanced(s):
    stack = []
    for letter in s:
        if letter == '{':
            stack.append(1)
        elif letter == '[':
            stack.append(2)
        elif letter == '(':
            stack.append(3)
        elif letter == '}':
            if len(stack) == 0:
                return False
            if stack.pop() != 1:
                return False
        elif letter == ']':
            if len(stack) == 0:
                return False
            if stack.pop() != 2:
                return False
        elif letter == ')':
            if len(stack) == 0:
                return False
            if stack.pop() != 3:
                return False
    return len(stack) == 0
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        s = input().strip()
        result = isBalanced(s)
        if result is True:
            print('YES')
        else:
            print('NO')
EOF
def nimGame(pile):
    res = reduce((lambda x, y: x ^ y), pile)
    if res == 0:
        return 'Second'
    else:
        return 'First'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    g = int(input())
    for g_itr in range(g):
        n = int(input())
        pile = list(map(int, input().rstrip().split()))
        result = nimGame(pile)
        fptr.write(result + '\n')
    fptr.close()
EOF
class Solution(object):
    def findLongestWord(self, s, d):
        d.sort(key = lambda x: (-len(x), x))
        for word in d:
            i = 0
            for c in s:
                if i < len(word) and word[i] == c:
                    i += 1
            if i == len(word):
                return word
        return ""
EOF
def isValid(s):
    cnt = Counter(s)
    res = 'NO'
    print("cnt = {} len = {}".format(cnt, len(set(cnt.values()))))
    if len(set(cnt.values())) == 1:
        res = 'YES'
    elif len(set(cnt.values())) == 2:
        bigger = max(cnt.values())
        lesser = min(cnt.values())
        bigger_let = [let for let, c in cnt.items() if c == bigger]
        lesser_let = [let for let, c in cnt.items() if c == lesser]
        if len(lesser_let) == 1 and lesser == 1:
            res = 'YES'
        elif len(bigger_let) == 1 or len(lesser_let) == 1:
            if abs(bigger-lesser) == 1:
                res = 'YES'
            else:
                res = 'NO'
        else:
            res = 'NO'
    else:
        res = 'NO'
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = isValid(s)
    fptr.write(result + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestPalindrome(self, s):
        longest = ""
        centres = [len(s) - 1]
        for diff in range(1, len(s)):  
            centres.append(centres[0] + diff)
            centres.append(centres[0] - diff)
        for centre in centres:
            if (min(centre + 1, 2 * len(s) - 1 - centre) <= len(longest)):
                break  
            if centre % 2 == 0:
                left, right = (centre // 2) - 1, (centre // 2) + 1
            else:
                left, right = centre // 2, (centre // 2) + 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            if right - left - 1 > len(longest):
                longest = s[left + 1:right]
        return longest
EOF
class Solution(object):
    def search(self, reader, target):
        left, right = 0, 19999
        while left <= right:
            mid = left + (right-left)//2
            response = reader.get(mid)
            if response > target:
                right = mid-1
            elif response < target:
                left = mid+1
            else:
                return mid
        return -1
EOF
    res = 0
    if bc <= wc:
        res += bc*b
        if bc + z <= wc:
            res += (bc + z)*w
        else:
            res += wc*w
    else:
        res += wc*w
        if wc + z <= bc:
            res += (wc + z)*b
        else:
            res += bc*b
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        bw = input().split()
        b = int(bw[0])
        w = int(bw[1])
        bcWcz = input().split()
        bc = int(bcWcz[0])
        wc = int(bcWcz[1])
        z = int(bcWcz[2])
        result = taumBday(b, w, bc, wc, z)
        fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def frequencySort(self, s):
        freq = Counter(s)
        pairs = [(count, c) for c, count in freq.items()]
        pairs.sort(reverse = True)
        result = []
        for count, c in pairs:
            result += [c] * count
        return "".join(result)
EOF
def findDigits(n):
    temp = n
    count = 0
    while temp:
        if temp % 10 > 0:
            count += 1 if not n%(temp%10) else 0
        temp //= 10
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        result = findDigits(n)
        fptr.write(str(result) + '\n')
    fptr.close()
EOF
def insertionSort1(start, arr):
    probe = arr[start]
    changed = 0
    for ind in range(start-1, -1, -1):
        if arr[ind] > probe:
            changed += 1
            arr[ind+1] = arr[ind]
        else:
            arr[ind+1] = probe
            break
    if arr[0] > probe:
        arr[0] = probe
    return changed
def insertionSort2(n, arr):
    res = 0
    for ind in range(1, len(arr)):
        res += insertionSort1(ind, arr)
    return res
def runningTime(arr):
    return insertionSort2(len(arr), arr)
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = runningTime(arr)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class RLEIterator(object):
    def __init__(self, A):
        self.encoding = A
        self.length = len(A)
        self.i = 0                  
    def next(self, n):
        while self.i < self.length and self.encoding[self.i] < n:   
            n -= self.encoding[self.i]      
            self.i += 2                     
        if self.i >= self.length:
            return -1
        self.encoding[self.i] -= n          
        return self.encoding[self.i + 1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sumEvenAfterQueries(self, A, queries):
        sum_even = sum(x for x in A if x % 2 == 0)
        result = []
        for val, i in queries:
            if A[i] % 2 == 0:
                sum_even -= A[i]
            A[i] += val
            if A[i] % 2 == 0:
                sum_even += A[i]
            result.append(sum_even)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        n = len(A)
        if n < 3:
            return 0
        diff = A[1] - A[0]
        start, slices = 0, 0  
        for i in range(2, n):
            next_diff = A[i] - A[i - 1]
            if next_diff == diff:
                slices += i - start - 1
            else:
                diff = next_diff
                start = i - 1
        return slices
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countBits(self, num):
        ones = [0]
        for i in range(1, num + 1):
            ones.append(1 + ones[i & (i - 1)])      
        return ones
EOF
cube = lambda x: x**3
def fibonacci(n):
    if n == 0:
        return []
    if n == 1:
        return [0]
    prev = 0
    cur = 1
    out = [prev, cur]
    for _ in range(n-2):
        prev, cur = cur, prev + cur
        out.append(cur)
    return out
EOF
if __name__ == "__main__":
    n = int(input().strip())
    array = []
    print_dict = {}
    for a0 in range(n):
        x, s = input().strip().split(' ')
        x, s = [int(x), str(s)]
        if a0 < n//2:
            array.append((x, "-"))
        else:
            array.append((x, s))
    print(" ".join(map(lambda x: x[1], sorted(array, key = lambda x: x[0]))))
EOF
class Solution(object):
    def canBeEqual(self, target, arr):
        return collections.Counter(target) == collections.Counter(arr)
class Solution2(object):
    def canBeEqual(self, target, arr):
        target.sort(), arr.sort()
        return target == arr
EOF
def is_beautiful(num, k):
    rev_num = int(str(num)[::-1])
    return abs(num - rev_num)%k == 0
def beautifulDays(i, j, k):
    res = 0
    for num in  range(i, j+1):
        if is_beautiful(num, k):
            res += 1
    return res
if __name__ == "__main__":
    i, j, k = raw_input().strip().split(' ')
    i, j, k = [int(i), int(j), int(k)]
    result = beautifulDays(i, j, k)
    print result
EOF
class Solution(object):
    def minTime(self, n, edges, hasApple):
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        result = [0, 0]
        s = [(1, (-1, 0, result))]
        while s:
            step, params = s.pop()
            if step == 1:
                par, node, ret = params
                ret[:] = [0, int(hasApple[node])]
                for nei in reversed(graph[node]):
                    if nei == par:
                        continue
                    new_ret = [0, 0]
                    s.append((2, (new_ret, ret)))
                    s.append((1, (node, nei, new_ret)))
            else:
                new_ret, ret = params
                ret[0] += new_ret[0]+new_ret[1]
                ret[1] |= bool(new_ret[0]+new_ret[1])
        return 2*result[0]
class Solution_Recu(object):
    def minTime(self, n, edges, hasApple):
        def dfs(graph, par, node, hasApple):
            result, extra = 0, int(hasApple[node])
            for nei in graph[node]:
                if nei == par:
                    continue
                count, found = dfs(graph, node, nei, hasApple)
                result += count+found
                extra |= bool(count+found)
            return result, extra
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        return 2*dfs(graph, -1, 0, hasApple)[0]
class Solution2(object):
    def minTime(self, n, edges, hasApple):
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        result = [0]
        s = [(1, (-1, 0, result))]
        while s:
            step, params = s.pop()
            if step == 1:
                par, node, ret = params
                tmp = [int(hasApple[node])]
                s.append((3, (tmp, ret)))
                for nei in reversed(graph[node]):
                    if nei == par:
                        continue
                    new_ret = [0]
                    s.append((2, (new_ret, tmp, ret)))
                    s.append((1, (node, nei, new_ret)))
            elif step == 2:
                new_ret, tmp, ret = params
                ret[0] += new_ret[0]
                tmp[0] |= bool(new_ret[0])
            else:
                tmp, ret = params
                ret[0] += tmp[0]
        return 2*max(result[0]-1, 0)
class Solution2_Recu(object):
    def minTime(self, n, edges, hasApple):
        def dfs(graph, par, node, has_subtree):
            result, extra = 0, int(hasApple[node])
            for nei in graph[node]:
                if nei == par:
                    continue
                count = dfs(graph, node, nei, hasApple)
                result += count
                extra |= bool(count)
            return result+extra
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        return 2*max(dfs(graph, -1, 0, hasApple)-1, 0)
EOF
def largestPermutation(k, arr):
    maxcur = max(arr)
    positions = {}
    for ind in range(len(arr)):
        positions[arr[ind]] = ind
    for ind in range(len(arr)):
        if k == 0:
            break
        if arr[ind] == maxcur:
            maxcur -= 1
        if arr[ind] < maxcur:
            mind = positions[maxcur]
            positions[maxcur], positions[arr[ind]] = positions[arr[ind]], positions[maxcur]
            arr[ind], arr[mind] = arr[mind], arr[ind]
            maxcur -= 1
            k -= 1
    return arr
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    arr = list(map(int, input().strip().split(' ')))
    result = largestPermutation(k, arr)
    print (" ".join(map(str, result)))
EOF
def is_leap_greg(year):
    return (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0))
def is_leap_jul(year):
    return year % 4 == 0
def solve(year):
    target = 256
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year < 1918:
        if is_leap_jul(year):
            months[1] = 29
    elif year > 1919:
        if is_leap_greg(year):
            months[1] = 29
    elif year == 1918:
        target += 13
    sum_tmp = 0
    for ind, el in enumerate(months):
        if sum_tmp + el > target:
            month = '0' + str(ind + 1)
            day = target - sum_tmp
            break
        else:
            sum_tmp += el
    return ".".join([str(day), str(month), str(year)])
year = int(input().strip())
result = solve(year)
print(result)
EOF
def repeatedString(s, n):
    return s.count('a') * (n//len(s)) + s[:n%len(s)].count('a')
if __name__ == "__main__":
    s = input().strip()
    n = int(input().strip())
    result = repeatedString(s, n)
    print(result)
EOF
class Solution(object):
    def isNStraightHand(self, hand, W):
        if len(hand) % W:
            return False
        counts = Counter(hand)
        min_heap = list(hand)
        heapify(min_heap)
        for _ in xrange(len(min_heap)//W):
            while counts[min_heap[0]] == 0:
                heappop(min_heap)
            start = heappop(min_heap)
            for _ in xrange(W):
                counts[start] -= 1
                if counts[start] < 0:
                    return False
                start += 1
        return True
EOF
def capitalize(string):
    for word in string.split():
        string = string.replace(word, word.capitalize())
    return string
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxChunksToSorted(self, arr):
        min_right = [float("inf") for _ in range(len(arr))]     
        for i in range(len(arr) - 2, -1, -1):
            min_right[i] = min(min_right[i + 1], arr[i + 1])
        partitions = 0
        partition_max = None
        for i, num in enumerate(arr):
            partition_max = num if partition_max is None else max(partition_max, num)
            if partition_max < min_right[i]:
                partitions += 1
                partition_max = None
        return partitions
EOF
class Solution(object):
    def constructRectangle(self, area):
        w = int(math.sqrt(area))
        while area % w:
            w -= 1
        return [area // w, w]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def powerfulIntegers(self, x, y, bound):
        result = set()
        def make_power_list(val):
            power_list = [1]
            if val != 1:            
                while power_list[-1] <= bound:
                    power_list.append(power_list[-1] * val)
                power_list.pop()    
            return power_list
        x_list, y_list = make_power_list(x), make_power_list(y)
        for x_num in x_list:
            for y_num in y_list:
                if x_num + y_num > bound:
                    break
                result.add(x_num + y_num)
        return list(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class NumArray(object):
    def __init__(self, nums):
        self.cumul = [0]
        for num in nums:
            self.cumul.append(self.cumul[-1] + num)
    def sumRange(self, i, j):
        return self.cumul[j + 1] - self.cumul[i]
EOF
def countSwaps(a):
    count = 0
    for i in range(len(a)):
        flag = 0
        for j in range(len(a)-i-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                count += 1
                flag = 1
        if flag == 0:
            break
    print("Array is sorted in",count,"swaps.")
    print("First Element:",a[0])
    print("Last Element:",a[-1])
if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().rstrip().split()))
    countSwaps(a)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def leastBricks(self, wall):
        edges = defaultdict(int)  
        for row in wall:
            edge = 0
            for brick in row:
                edge += brick
                edges[edge] += 1
        del edges[sum(wall[0])]  
        crossed = len(wall)  
        return crossed if not edges else crossed - max(edges.values())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def mergeStones(self, stones, K):
        n = len(stones)
        if (n - 1) % (K - 1) != 0:  
            return -1
        prefix_sum = [0] * (n + 1)  
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + stones[i]
                def helper(i, j):           
            if j - i + 1 < K:       
                return 0
            res = min(helper(i, mid) + helper(mid + 1, j) for mid in range(i, j, K - 1))
            if (j - i) % (K - 1) == 0:
                res += prefix_sum[j + 1] - prefix_sum[i]
            return res
        return helper(0, n - 1)
EOF
def change(match):
    symb = match.group(0)
    if symb == "&&":
        return "and"
    elif symb == "||":
        return "or"
n = int(input().strip())
for _ in range(n):
    print(re.sub(r'(?<= )(&&|\|\|)(?= )', change, input()))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def replaceWords(self, dict, sentence):
        result = []
        root = {}
        for word in dict:
            node = root
            for c in word[:-1]:
                if c not in node:   
                    node[c] = {}
                elif isinstance(node[c], str):  
                    break
                node = node[c]
            else:
                node[word[-1]] = word           
        sentence = sentence.split(" ")
        for word in sentence:
            node = root
            for c in word:
                if c not in node:
                    result.append(word)         
                    break
                if isinstance(node[c], str):    
                    result.append(node[c])
                    break
                node = node[c]
            else:
                result.append(word)
        return " ".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isStrobogrammatic(self, num):
        strob = {'0':'0', '1':'1', '8':'8', '6':'9', '9':'6'}       
        for left in range((len(num) + 1) // 2):                     
            right = len(num) - 1 - left
            if num[left] not in strob or strob[num[left]] != num[right]:
                return False
        return True
EOF
class Solution(object):
    def reverseWords(self, s):
        return ' '.join(reversed(s.split()))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def strobogrammaticInRange(self, low, high):
        max_len, min_len = len(high), len(low)
        low, high = int(low), int(high)
        live_list = ['']                
        other_list = ['0', '1', '8']    
        strobo_count = 0
        strobo = {'0' : '0', '1' : '1', '8': '8', '6' : '9', '9' : '6'}
        if min_len == 1:
            strobo_count += len([i for i in other_list if low <= int(i) <= high])
        for i in range(2, max_len+1):
            live_list = [c + r + strobo[c] for r in live_list for c in strobo]  
            if min_len < i < max_len:           
                strobo_count += len([True for result in live_list if result[0] != '0'])
            elif i == min_len or i == max_len:  
                strobo_count += len([True for result in live_list if result[0] != '0' and low <= int(result) <= high])
            live_list, other_list = other_list, live_list   
        return strobo_count
EOF
n = int(input().strip())
inside = False
for _ in range(n):
    line = input()
    for el in line.split(' '):
        if el == "{":
            inside = True
            continue
        elif el == "}":
            inside = False
            continue
        elif inside:
            found = re.search(r'\
            if found:
                print(found.group(0))
EOF
class Solution(object):
    def getFactors(self, n):
        result = []
        factors = []
        self.getResult(n, result, factors)
        return result
    def getResult(self, n, result, factors):
        i = 2 if not factors else factors[-1]
        while i <= n / i:
            if n % i == 0:
                factors.append(i)
                factors.append(n / i)
                result.append(list(factors))
                factors.pop()
                self.getResult(n / i, result, factors)
                factors.pop()
            i += 1
EOF
def getMinimumCost(k, flowers):
    res = 0
    cnt = 0
    flowers = sorted(flowers, key=lambda x: -x)
    for el in flowers:
        res += el * (1 + cnt//k)
        cnt += 1
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    c = list(map(int, input().rstrip().split()))
    minimumCost = getMinimumCost(k, c)
    fptr.write(str(minimumCost) + '\n')
    fptr.close()
EOF
def findMedian(arr):
    arr = sorted(arr)
    return arr[len(arr)//2]
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = findMedian(arr)
    print(result)
EOF
def andProduct(a, b):
    if a == 0:
        return 0
    if int(log2(a)) != int(log2(b)):
        return 0
    else:
        res_bin = list(bin(a)[2:])
        len_res = len(res_bin)
        for ind, digit in enumerate(res_bin):
            if ind == 0 or digit == '0':
                continue
            else:
                test_bin = list(bin(b)[2:])
                test_bin[ind] = '0'
                test_dec = int("".join(test_bin), 2)
                if a <= test_dec <= b:
                    res_bin[ind] = '0'
        return int("".join(res_bin), 2)
if __name__ == "__main__":
    n = int(input().strip())
    for a0 in range(n):
        a, b = input().strip().split(' ')
        a, b = [int(a), int(b)]
        result = andProduct(a, b)
        print(result)
EOF
class BinaryMatrix(object):
    def get(self, row, col):
        pass
    def dimensions(self):
        pass
class Solution(object):
    def leftMostColumnWithOne(self, binaryMatrix):
        m, n = binaryMatrix.dimensions()
        r, c = 0, n-1
        while r < m and c >= 0:
            if not binaryMatrix.get(r, c):
                r += 1
            else:
                c -= 1        
        return c+1 if c+1 != n else -1
EOF
def jimOrders(orders):
    sequence = []
    for ind, el in enumerate(orders, 1):
        time = sum(el)
        insort(sequence, (time, ind))
    return list(map(lambda x: x[1], sequence))
if __name__ == "__main__":
    n = int(input().strip())
    orders = []
    for orders_i in range(n):
        orders_t = [int(orders_temp) for orders_temp in input().strip().split(' ')]
        orders.append(orders_t)
    result = jimOrders(orders)
    print (" ".join(map(str, result)))
EOF
class Solution(object):
    def findLongestChain(self, pairs):
        pairs.sort(key=lambda x: x[1])
        cnt, i = 0, 0
        for j in xrange(len(pairs)):
            if j == 0 or pairs[i][1] < pairs[j][0]:
                cnt += 1
                i = j
        return cnt
EOF
def timeInWords(h, m):
    res = ''
    numbers = ['','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve', 'thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty']
    minute = 'minute'
    if m != 1:
        minute += 's'
    if m == 0:
        res = numbers[h] + " o' clock"
    elif m == 30:
        res = "half past " + numbers[h]
    elif m == 15:
        res = "quarter past " + numbers[h]
    elif m == 45:
        res = "quarter to " + numbers[h + 1]
    elif m < 20:
        res = numbers[m] + ' ' + minute + ' past ' + numbers[h]
    elif m < 30:
        res = numbers[-1] + ' ' + numbers[int(m%10)] + ' ' + minute + ' past ' + numbers[h]
    elif m > 45:
        res = numbers[60 - m] + ' ' + minute + ' to ' + numbers[h + 1]
    elif m > 30:
        res = numbers[-1] + ' ' + numbers[int(m%10)] + ' ' + minute + ' to ' + numbers[h + 1]
    return res.replace('  ', ' ')
if __name__ == "__main__":
    h = int(input().strip())
    m = int(input().strip())
    result = timeInWords(h, m)
    print(result)
EOF
num = input()
print(bool(re.match(r'^[1-9][\d]{5}$', num) and len(re.findall(r'(\d)(?=\d\1)', num))<2 ))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def subdomainVisits(self, cpdomains):
        counts = defaultdict(int)                   
        for cpdomain in cpdomains:
            count, domains = cpdomain.split(" ")    
            domains = domains.split(".")            
            for i in range(len(domains)):
                domain = ".".join(domains[i:])      
                counts[domain] += int(count)        
        return [str(count) + " " + domain for domain, count in counts.items()]  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canBreak(self, s, wordDict):
        can_make = [False] * (len(s)+1)         
        can_make[0] = True
        for i in range(1, len(s)+1):            
            for j in range(i-1, -1, -1):        
                if can_make[j] and s[j:i] in wordDict:
                    can_make[i] = True
                    break
        return can_make[-1]
    def wordBreak(self, s, wordDict):
        if not self.canBreak(s, wordDict):
            return []
        result_lists = self.break_word(s, 0, wordDict, {})
        return [" ".join(result) for result in result_lists]    
    def break_word(self, s, left, wordDict, memo):      
        if left >= len(s):      
            return [[]]
        if left in memo:
            return memo[left]
        results = []
        for i in range(left+1, len(s)+1):       
            prefix = s[left:i]
            suffix_breaks = self.break_word(s, i, wordDict, memo)
            if suffix_breaks and prefix in wordDict:
                for suffix_break in suffix_breaks:
                    results.append([prefix] + suffix_break)
        memo[left] = results[:]
        return results
EOF
def luckBalance(n, k, arr):
    res = sum(list(map(lambda x: x[0], filter(lambda x: x[1] == 0, arr))))
    arr = sorted(arr, key=lambda x: (-x[1], -x[0]))
    important = len(list(filter(lambda x: x[1] == 1, arr)))
    kcnt = 0
    for ind in range(important):
        if kcnt < k:
            res += arr[ind][0]
            kcnt += 1
        else:
            res -= arr[ind][0]
    return res
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    contests = []
    for contests_i in range(n):
        contests_t = [int(contests_temp) for contests_temp in input().strip().split(' ')]
        contests.append(contests_t)
    result = luckBalance(n, k, contests)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyStack(object):
    def __init__(self):
        self.queue = deque()
    def push(self, x):
        self.queue.appendleft(x)
    def pop(self):
        new_queue = deque()
        while True:
            x = self.queue.pop()
            if not self.queue:
                self.queue = new_queue
                return x
            new_queue.appendleft(x)
    def top(self):
        new_queue = deque()
        while self.queue:
            x = self.queue.pop()
            new_queue.appendleft(x)
        self.queue = new_queue
        return x
    def empty(self):
        return len(self.queue) == 0
EOF
def solveMeFirst(a,b):
    return a + b
num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def printLinkedList(head):
    cur = head
    while cur:
        print(cur.data)
        cur = cur.next
if __name__ == '__main__':
    llist_count = int(input())
    llist = SinglyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)
    printLinkedList(llist.head)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findContentChildren(self, g, s):
        content = 0         
        child = 0           
        g.sort()
        s.sort()
        for cookie in s:
            if child == len(g): 
                break
            if g[child] <= cookie:  
                content += 1
                child += 1
        return content
EOF
def is_kaprekar(num):
    if num < 4:
        if num == 1:
            return True
        else:
            return False
    num_sq = pow(num, 2)
    num_sq_str = str(num_sq)
    left = num_sq_str[:len(num_sq_str)//2]
    right = num_sq_str[len(num_sq_str)//2:]
    if int(left) + int(right) == num:
        return True
    else:
        return False
def kaprekarNumbers(p, q):
    out = []
    for num in range(p, q + 1):
        if is_kaprekar(num):
            out.append(num)
    return out
if __name__ == "__main__":
    p = int(input().strip())
    q = int(input().strip())
    result = kaprekarNumbers(p, q)
    if result:
        print (" ".join(map(str, result)))
    else:
        print("INVALID RANGE")
EOF
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def constructFromPrePost(self, pre, post):
        stack = [TreeNode(pre[0])]
        j = 0
        for i in xrange(1, len(pre)):
            node = TreeNode(pre[i])
            while stack[-1].val == post[j]:
                stack.pop()
                j += 1
            if not stack[-1].left:
                stack[-1].left = node
            else:
                stack[-1].right = node
            stack.append(node)
        return stack[0]
class Solution2(object):
    def constructFromPrePost(self, pre, post):
        def constructFromPrePostHelper(pre, pre_s, pre_e, post, post_s, post_e, post_entry_idx_map):
            if pre_s >= pre_e or post_s >= post_e:
                return None
            node = TreeNode(pre[pre_s])
            if pre_e-pre_s > 1:
                left_tree_size = post_entry_idx_map[pre[pre_s+1]]-post_s+1
                node.left = constructFromPrePostHelper(pre, pre_s+1, pre_s+1+left_tree_size, 
                                                       post, post_s, post_s+left_tree_size,
                                                       post_entry_idx_map)
                node.right = constructFromPrePostHelper(pre, pre_s+1+left_tree_size, pre_e,
                                                        post, post_s+left_tree_size, post_e-1,
                                                        post_entry_idx_map)
            return node
        post_entry_idx_map = {}
        for i, val in enumerate(post):
            post_entry_idx_map[val] = i
        return constructFromPrePostHelper(pre, 0, len(pre), post, 0, len(post), post_entry_idx_map)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def arrayPairSum(self, nums):
        return sum(sorted(nums)[::2])   
EOF
class Solution(object):
    def shortestCompletingWord(self, licensePlate, words):
        def contains(counter1, w2):
            c2 = collections.Counter(w2.lower())
            c2.subtract(counter1)
            return all(map(lambda x: x >= 0, c2.values()))
        result = None
        counter = collections.Counter(c.lower() for c in licensePlate if c.isalpha())
        for word in words:
            if (result is None or (len(word) < len(result))) and \
               contains(counter, word):
                result = word
        return result
EOF
class Cell:
    def __init__(self, x, y, dist):
        self.x = x
        self.y = y
        self.dist = dist
def point_is_valid(horse, size):
    if any(x < 1 for x in horse) or any(x > size for x in horse):
        return False
    else:
        return True
def print_path(init, parent):
    tup = (init[0], init[1])
    if tup in parent.keys():
        prev = parent[tup]
        print_path((prev[0], prev[1]), parent)
    print(tup, end=' ')
def find_path_dijkstra(horse, king, size, a, b):
    res_dist = -1
    next_to_visit = {(horse[0], horse[1]):0}
    distances = [[-1] * (size + 1) for _ in range(size + 1)]
    distances[horse[0]][horse[1]] = 0
    parent = {}
    directions = [[a, b], [b, a], [a, -b], [b, -a], [-a, b], [-b, a], [-a, -b], [-b, -a]]
    while bool(next_to_visit):
        c = min(next_to_visit, key=next_to_visit.get)
        del next_to_visit[c]
        if c[0] == king[0] and c[1] == king[1]:
            res_dist = distances[c[0]][c[1]]
            break
        for di in directions:
            x, y = c[0] + di[0], c[1] + di[1]
            if point_is_valid([x, y], size) and distances[x][y] == -1:
                temp_path = distances[c[0]][c[1]] + 1
                if distances[x][y] == -1 or distances[x][y] > temp_path:
                    distances[x][y] = temp_path
                    next_to_visit[(x, y)] = temp_path
                    parent[(x, y)] = [c[0], c[1]]
    return res_dist
if __name__ == "__main__":
    board_size = int(input().strip())
    horse = [1, 1]
    king = [board_size, board_size]
    for a in range(1, board_size):
        for b in range(1, board_size):
            print("{}".format(find_path_dijkstra(horse, king, board_size, a, b)), end = ' ')
        print()
EOF
class Solution(object):
    def mySqrt(self, x):
        if x < 2:
            return x
        left, right = 1, x // 2
        while left <= right:
            mid = left + (right - left) // 2
            if mid > x / mid:
                right = mid - 1
            else:
                left = mid + 1
        return left - 1
EOF
def dec_to_bin(number):
    return bin(number)[2:]
def bin_to_dec(number):
    return int(number, 2)
def buildup_to_len(number, length):
    return '0' * (length - len(number)) + number
def get_max_len(arr):
    return len(dec_to_bin(max(arr)))
def get_min_len(arr):
    return len(dec_to_bin(min(arr)))
def anotherMinimaxProblem(a):
    res = inf
    arr_bin = []
    if max(a) == min(a) and max(a) == 0:
        return 0
    for el in a:
        arr_bin.append(dec_to_bin(el))
    while len(max(arr_bin, key=len)) == len(min(arr_bin, key=len)):
        arr_bin = [ el[:1].lstrip('1') + el[1:] for el in arr_bin ]
        arr_bin = [ el.lstrip('0') for el in arr_bin ]
    max_len = len(max(arr_bin, key=len))
    arr_bin = [ buildup_to_len(el, max_len) for el in arr_bin ]
    arr_zeros = []
    arr_ones = []
    for el in sorted(arr_bin):
        if el[0] == '0':
            arr_zeros.append(el)
        else:
            arr_ones.append(el)
    for el_z in arr_zeros:
        for el_o in arr_ones:
            res = min(res, bin_to_dec(el_z) ^ bin_to_dec(el_o))
    return res
if __name__ == "__main__":
    n = int(input().strip())
    a = list(map(int, input().strip().split(' ')))
    result = anotherMinimaxProblem(a)
    print(result)
EOF
def find_adjacent(grid, i, j):
    count = 0
    if i < 0 or j < 0 or i >= n or j >= m:
        return 0
    if grid[i][j] == 0:
        return 0
    count += 1
    grid[i][j] -= 1
    count += find_adjacent(grid, i-1, j-1)
    count += find_adjacent(grid, i-1, j)
    count += find_adjacent(grid, i-1, j+1)
    count += find_adjacent(grid, i+1, j-1)
    count += find_adjacent(grid, i+1, j)
    count += find_adjacent(grid, i+1, j+1)
    count += find_adjacent(grid, i, j-1)
    count += find_adjacent(grid, i, j+1)
    return count
def get_biggest_region(grid):
    biggest = 0
    for i in range(n):
        for j in range(m):
            result = find_adjacent(grid, i, j)
            if result > biggest:
                biggest = result
    return biggest
n = int(input().strip())
m = int(input().strip())
grid = []
for grid_i in range(n):
    grid_t = list(map(int, input().strip().split(' ')))
    grid.append(grid_t)
print(get_biggest_region(grid))
EOF
class FooBar(object):
    def __init__(self, n):
        self.__n = n
        self.__curr = False
        self.__cv = threading.Condition()
    def foo(self, printFoo):
        for i in xrange(self.__n):
            with self.__cv:
                while self.__curr != False:
                    self.__cv.wait()
                self.__curr = not self.__curr
                printFoo()
                self.__cv.notify()
    def bar(self, printBar):
        for i in xrange(self.__n):
            with self.__cv:
                while self.__curr != True:
                        self.__cv.wait()
                self.__curr = not self.__curr
                printBar()
                self.__cv.notify()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def nthMagicalNumber(self, N, A, B):
        low, high = 1, 10 ** 14
        def gcd(a, b):
            a, b = b, a % b
            if b == 0:
                return a
            return gcd(a, b)
        lcm = A * B // gcd(A, B)
        while low < high:
            mid = (low + high) // 2
            num = (mid // A) + (mid // B) - (mid // lcm)    
            if num < N:                                     
                low = mid + 1
            elif num >= N:                                  
                high = mid
        return low % (10 ** 9 + 7)
EOF
class Node:
    def __init__(self, info): 
        self.info = info  
        self.left = None  
        self.right = None 
        self.level = None 
    def __str__(self):
        return str(self.info) 
class BinarySearchTree:
    def __init__(self): 
        self.root = None
    def create(self, val):  
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break
def postOrder(root):
    if not root:
        return None
    postOrder(root.left)
    postOrder(root.right)
    print(root.info,end=" ")
tree = BinarySearchTree()
t = int(input())
arr = list(map(int, input().split()))
for i in range(t):
    tree.create(arr[i])
postOrder(tree.root)
EOF
input_string = input()
print('Hello, World.');
print (input_string);
EOF
class Node:
    def __init__(self,data):
        self.data = data
        self.next = None 
class Solution: 
    def display(self,head):
        current = head
        while current:
            print(current.data,end=' ')
            current = current.next
    def insert(self,head,data): 
        if head == None:
            head = Node(data)
        else:
            cur = head
            while cur.next != None:
                cur = cur.next
            cur.next = Node(data)
        return head
mylist= Solution()
T=int(input())
head=None
for i in range(T):
    data=int(input())
    head=mylist.insert(head,data)    
mylist.display(head); 	  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        subarrays = 0
        start = 0           
        product = 1         
        for end, num in enumerate(nums):
            product *= num
            while product >= k and start <= end:    
                product //= nums[start]
                start += 1
            subarrays += end - start + 1            
        return subarrays
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def verifyPreorder(self, preorder):
        stack = [float('inf')]      
        minimum = float('-inf')
        for value in preorder:
            if value < minimum:
                return False
            while value > stack[-1]:
                minimum = stack.pop()   
            stack.append(value)
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])
        min_path = [float('inf') for _ in range(n + 1)]
        min_path[1] = 0
        for row in range(1, m + 1):
            new_min_path = [float('inf') for _ in range(n + 1)]
            for col in range(1, n + 1):
                new_min_path[col] = grid[row - 1][col - 1] + min(min_path[col], new_min_path[col - 1])
            min_path = new_min_path
        return min_path[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLengthOfLCIS(self, nums):
        longest, current = 0, 0
        for i, num in enumerate(nums):
            if i == 0 or num <= nums[i - 1]:
                current = 0
            current += 1
            longest = max(longest, current)
        return longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def levelOrderBottom(self, root):
        traversal = []
        self.inorder(root, 0, traversal)
        return traversal[::-1]
    def inorder(self, node, depth, traversal):
        if not node:
            return
        if len(traversal) == depth:
            traversal.append([])
        self.inorder(node.left, depth+1, traversal)
        traversal[depth].append(node.val)
        self.inorder(node.right, depth+1, traversal)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def buddyStrings(self, A, B):
        if len(A) != len(B):
            return False
        if A == B:
            return any(count > 1 for count in Counter(A).values())
        diffs = [i for i in range(len(A)) if A[i] != B[i]]      
        if len(diffs) != 2:
            return False
        return A[diffs[0]] == B[diffs[1]] and A[diffs[1]] == B[diffs[0]] 
EOF
time = input().strip()
hh = time[0:2]
period = time[-2:]
if period.lower() == "am":
    if hh == "12":
        t = "00" + time[2:-2]
    else:
        t = time[0:-2]
else:
    if hh == "12":
        t = time[0:-2]
    else:
        t = str(int(hh) + 12) + time[2:-2]
print(t)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if not m or not n:
            return 0
        if obstacleGrid[0][0] or obstacleGrid[-1][-1]:
            return 0
        row_paths = [0 for _ in range(n+1)]
        row_paths[1] = 1
        for row in range(1, m+1):
            new_row_paths = [0]
            for col in range(1, n+1):
                if obstacleGrid[row-1][col-1]:
                    new_row_paths.append(0)
                else:
                    new_row_paths.append(new_row_paths[-1] + row_paths[col])
            row_paths = new_row_paths
        return row_paths[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findIntegers(self, num):
        binary = bin(num)[2:][::-1]
        zero_highest = [1]
        one_highest = [1]
        if binary[0] == "0":
            count = 1
        else:
            count = 2
        for bit in range(1, len(binary)):  
            zero_highest.append(zero_highest[-1] + one_highest[-1])
            one_highest.append(zero_highest[-2])
            if binary[bit] == "1" and binary[bit - 1] == "1":
                count = zero_highest[-1] + one_highest[-1]
            elif binary[bit] == "1" and binary[bit - 1] == "0":
                count += zero_highest[-1]
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxWidthRamp(self, A):
        max_ramp = 0
        stack = []                              
        for i, num in enumerate(A):
            if not stack or num < A[stack[-1]]:
                stack.append(i)
        for i in range(len(A) - 1, -1, -1):     
            while stack and A[i] >= A[stack[-1]]:
                max_ramp = max(max_ramp, i - stack.pop())
        return max_ramp
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def removeNthFromEnd(self, head, n):
        first, second = head, head
        for i in range(n):      
            first = first.next
        if not first:
            return head.next
        while first.next:       
            first = first.next
            second = second.next
        second.next = second.next.next  
        return head
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reverseKGroup(self, head, k):
        if k < 2:
            return head
        node = head
        for _ in range(k):
            if not node:
                return head     
            node = node.next
        prev = self.reverseKGroup(node, k)
        for _ in range(k):      
            temp = head.next
            head.next = prev
            prev = head
            head = temp
        return prev
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isPowerOfFour(self, num):
        if num <= 0:
            return False
        return 4 ** int(math.log(num, 4)) == num
EOF
n = list(map(int, input().split()))
print(np.zeros(n, dtype=np.int))
print(np.ones(n, dtype=np.int))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def solveEquation(self, equation):
        x, val = 0, 0   
        base = 1        
        i = 0
        while i < len(equation):
            neg = base
            if equation[i] == "+":
                i += 1
            if equation[i] == "-":
                neg = -base
                i += 1
            num = None
            while i < len(equation) and "0" <= equation[i] <= "9":
                if num is None:
                    num = 0
                num = num * 10 + int(equation[i])
                i += 1
            if num is None:
                num = 1
            if i < len(equation) and equation[i] == "x":
                x += num * neg
                i += 1
            else:
                val += num * neg
            if i < len(equation) and equation[i] == "=":
                base *= -1
                i += 1
        if x == 0 and val == 0:
            return "Infinite solutions"
        if x == 0:
            return "No solution"
        return "x=" + str(-val // x)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def soupServings(self, N):
        memo = {}
        def helper(A, B):
            if A <= 0 and B <= 0:       
                return 0.5
            if A <= 0:                  
                return 1
            if B <= 0:                  
                return 0
            if (A, B) in memo:
                return memo[(A, B)]
            result = 0.25 * (helper(A - 4, B) + helper(A - 3, B - 1) + helper(A - 2, B - 2) + helper(A - 1, B - 3))
            memo[(A, B)] = result
            return result
        portions = math.ceil(N / float(25))
        if N > 4800:                    
        return helper(portions, portions)
EOF
if __name__ == "__main__":
    meal_cost = float(input().strip())
    tip_percent = int(input().strip())
    tax_percent = int(input().strip())
    tip = (tip_percent / 100) * meal_cost
    tax = (tax_percent / 100) * meal_cost
    total_cost = round(meal_cost + tip + tax)
    print("The total meal cost is " + str(total_cost) + " dollars.")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findTilt(self, root):
        self.tilt = 0       
        def helper(node):
            if not node:    
                return 0
            left, right = helper(node.left), helper(node.right)
            self.tilt += abs(left - right)  
            return node.val + left + right  
        helper(root)
        return self.tilt
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def characterReplacement(self, s, k):
        longest, start = 0, 0
        freq = defaultdict(int)
        for end in range(len(s)):
            freq[s[end]] += 1
            while (end - start + 1) - max(freq.values()) > k:
                freq[s[start]] -= 1
                start += 1
            longest = max(longest, end - start + 1)
        return longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maximumProduct(self, nums):
        nums.sort()
        top_3 = nums[-1] * nums[-2] * nums[-3]
        top_bottom = nums[-1] * nums[0] * nums[1]
        return max(top_3, top_bottom)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findClosestLeaf(self, root, k):
        nearest_leaves = {0: (float("inf"), 0)} 
        def closest_down(node):                 
            if not node:
                return (float("inf"), 0)
            if not node.left and not node.right:    
                result = (0, node.val)
            else:
                left_dist, left_nearest = closest_down(node.left)
                right_dist, right_nearest = closest_down(node.right)
                if left_dist <= right_dist:     
                    result = (left_dist + 1, left_nearest)
                else:
                    result = (right_dist + 1, right_nearest)
            nearest_leaves[node.val] = result
            return result
        def closest(node, parent_val):          
            if not node:
                return
            if 1 + nearest_leaves[parent_val][0] < nearest_leaves[node.val][0]:     
                nearest_leaves[node.val] = (1 + nearest_leaves[parent_val][0], nearest_leaves[parent_val][1])
            closest(node.left, node.val)
            closest(node.right, node.val)
        closest_down(root)
        closest(root, 0)
        return nearest_leaves[k][1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isPossible(self, nums):
        freq = Counter(nums)
        sequences = defaultdict(int)  
        for num in nums:
            if freq[num] == 0:  
                continue
            freq[num] -= 1
            if sequences[num - 1] != 0:  
                sequences[num - 1] -= 1
                sequences[num] += 1
            elif freq[num + 1] > 0 and freq[num + 2] > 0:  
                freq[num + 1] -= 1
                freq[num + 2] -= 1
                sequences[num + 2] += 1
            else:  
                return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wordPatternMatch(self, pattern, str):
        m, n = len(pattern), len(str)
        def is_match(i, j):
            if i >= m and j >= n:
                return True
            if i >= m:                              
                return False
            for end in range(j, n - (m - i) + 1):   
                p, test_s = pattern[i], str[j:end + 1]  
                if p not in mapping and test_s not in s_used:   
                    mapping[p] = test_s
                    s_used.add(test_s)
                    if is_match(i + 1, end + 1):
                        return True
                    del mapping[p]                  
                    s_used.discard(test_s)
                elif p in mapping and mapping[p] == test_s:     
                    if is_match(i + 1, end + 1):
                        return True
            return False
        mapping = {}
        s_used = set()
        return is_match(0, 0)
EOF
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    result = factorial(n)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
def simpleArraySum(ar):
    sum = 0
    for i in range(len(ar)):
        sum += ar[i]
    return sum
if __name__ == '__main__':
    ar_count = int(input())
    ar = list(map(int,input().split()))
    result = simpleArraySum(ar)
    print(result)
EOF
if __name__ == '__main__':
    students=[[input(), float(input())] for _ in range(int(input()))]
    c = sorted(set(grade for name,grade in students))[1]
    student_names =[name for name,grade in students if grade == c]
    print('\n'.join(sorted(student_names)))
EOF
if __name__ == '__main__':
    N = int(input())
    if (N % 2 != 0) or (N%2 == 0 and N >= 6 and N <= 20):
        print("Weird")
    else:
        print("Not Weird")
EOF
def calculate_days(day, cnt):
    res = 0
    for el in cnt.items():
        res += el[1] * (day // el[0])
    return res
def minTime(machines, goal):
    cnt = Counter(machines)
    day_limit = 10**13
    curgoal = 0
    curday = 0
    res = 0
    prev_big = day_limit
    prev_low = 0
    curday = day_limit // 2
    while True:
        if curday == prev_big or curday == prev_low:
            prev_low_res = calculate_days(prev_low, cnt)
            prev_big_res = calculate_days(prev_big, cnt)
            if prev_low_res >= goal:
                return prev_low
            else:
                return prev_big
        curgoal = calculate_days(curday, cnt)
        if curgoal < goal:
            prev_low = curday
            curday = curday + (1 + day_limit - curday)//2
            res = curday
        else:
            prev_big = curday
            day_limit = curday
            curday = curday // 2
            res = curday
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nGoal = input().split()
    n = int(nGoal[0])
    goal = int(nGoal[1])
    machines = list(map(int, input().rstrip().split()))
    ans = minTime(machines, goal)
    fptr.write(str(ans) + '\n')
    fptr.close()
EOF
class Solution(object):
    def exist(self, board, word):
        visited = [[False for j in xrange(len(board[0]))] for i in xrange(len(board))]
        for i in xrange(len(board)):
            for j in xrange(len(board[0])):
                if self.existRecu(board, word, 0, i, j, visited):
                    return True
        return False
    def existRecu(self, board, word, cur, i, j, visited):
        if cur == len(word):
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or visited[i][j] or board[i][j] != word[cur]:
            return False
        visited[i][j] = True
        result = self.existRecu(board, word, cur + 1, i + 1, j, visited) or\
                 self.existRecu(board, word, cur + 1, i - 1, j, visited) or\
                 self.existRecu(board, word, cur + 1, i, j + 1, visited) or\
                 self.existRecu(board, word, cur + 1, i, j - 1, visited)
        visited[i][j] = False
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def mySqrt(self, x):
        guess = x
        while guess * guess > x:
            guess = (guess + x // guess) // 2
        return guess
EOF
class Solution(object):
    def reverseStr(self, s, k):
        s = list(s)
        for i in xrange(0, len(s), 2*k):
            s[i:i+k] = reversed(s[i:i+k])
        return "".join(s)
EOF
def simpleArraySum(n, ar):
    sum = 0
    for elem in ar:
        sum += elem
    return sum
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = simpleArraySum(n, ar)
print(result)
EOF
class Node():
    def __init__(self, left=None, right=None, data=None):
        self.left = left
        self.right = right
        self.data = data
def find_node_recursive(root, data):
    if root is not None:
        res = None
        if root.data == data:
            res = root
        if res == None:
            res = find_node(root.left, data)
        if res == None:
            res = find_node(root.right, data)
        return res
def find_node(root, data):
    current = root 
    s = []
    done = 0
    while(not done):
        if current is not None:
            s.append(current)
            current = current.left 
        else:
            if(len(s) > 0):
                current = s.pop()
                if current.data == data:
                    return current
                current = current.right 
            else:
                done = 1
    return None
def height(root):
    if root is not None:
        return max(1 + height(root.left), 1 + height(root.right))
    else:
        return 0
def inOrder(root):
    if root is not None:
        inOrder(root.left)
        print("{} ".format(root.data), end='')
        inOrder(root.right)
def swap_level(root, level):
    if root is not None:
        if level == 1:
            root = swap_subtrees(root)
        else:
            swap_level(root.left, level - 1)
            swap_level(root.right, level - 1)
def swap_subtrees(node):
    node.left, node.right = node.right, node.left
    return node
if __name__ == "__main__":
    sys.setrecursionlimit(15000)
    root = Node(data = 1)
    node = root
    n = int(input().strip())
    for ind in range(1, n+1):
        a, b = map(int,input().split(' '))
        node = find_node(root, ind)
        if a != -1:
            node.left = Node(data = a)
        if b != -1:
            node.right = Node(data = b)
    t = int(input().strip())
    for ind in range(t):
        k = int(input().strip())
        for level in range(k, height(root)+1, k):
            swap_level(root, level)
        inOrder(root)
        print()
EOF
    base = math.log2((t + 2)/3)
    val_top = 3*2**(math.floor(base))
    print("base = {} base_t = {}".format(base, val_top))
    return val_top - (t - (val_top-2))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    result = strangeCounter(t)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
class Solution(object):
    def guessNumber(self, n):
        left, right = 1, n
        while left <= right:
            mid = left + (right - left) / 2
            if guess(mid) <= 0: 
                right = mid - 1
            else:
                left = mid + 1
        return left
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def advantageCount(self, A, B):
        B_i = sorted([(b, i) for i, b in enumerate(B)])
        result = [None] * len(A)
        i = 0
        for a in sorted(A):
            if a > B_i[i][0]:
                result[B_i[i][1]] = a
                i += 1
            else:
                result[B_i.pop()[1]] = a
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def smallestGoodBase(self, n):
        n = int(n)
        for max_power in range(int(math.log(n, 2)), 1, -1):     
            base = int(n ** max_power ** -1)                    
            if n == ((base ** (max_power + 1)) - 1) // (base - 1):      
                return str(base)
        return str(n - 1)   
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def leastInterval(self, tasks, n):
        counts = Counter(tasks)
        max_count = max(counts.values())
        result = (max_count - 1) * (n + 1)  
        for count in counts.values():
            if count == max_count:          
                result += 1
        return max(result, len(tasks))      
EOF
def find_adjacent(grid, i, j):
    count = 0
    if i < 0 or j < 0 or i >= n or j >= m:
        return 0
    if grid[i][j] == 0:
        return 0
    count += 1
    grid[i][j] -= 1
    count += find_adjacent(grid, i-1, j-1)
    count += find_adjacent(grid, i-1, j)
    count += find_adjacent(grid, i-1, j+1)
    count += find_adjacent(grid, i+1, j-1)
    count += find_adjacent(grid, i+1, j)
    count += find_adjacent(grid, i+1, j+1)
    count += find_adjacent(grid, i, j-1)
    count += find_adjacent(grid, i, j+1)
    return count
def get_biggest_region(grid):
    biggest = 0
    for i in range(n):
        for j in range(m):
            result = find_adjacent(grid, i, j)
            if result > biggest:
                biggest = result
    return biggest
n = int(input().strip())
m = int(input().strip())
grid = []
for grid_i in range(n):
    grid_t = list(map(int, input().strip().split(' ')))
    grid.append(grid_t)
print(get_biggest_region(grid))
EOF
def gradingStudents(grades):
    res = []
    for grade in grades:
        difference = 5 - (grade%5)
        if difference < 3 and grade >= 38:
            res.append(grade+difference)
        else:
            res.append(grade)
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    grades_count = int(input().strip())
    grades = []
    for _ in range(grades_count):
        grades_item = int(input().strip())
        grades.append(grades_item)
    result = gradingStudents(grades)
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def smallestRangeII(self, A, K):
        A = sorted(A)
        result = A[-1] - A[0]       
        left_min = A[0] + K         
        right_max = A[-1] - K
        for i in range(len(A) - 1):  
            lower = min(left_min, A[i + 1] - K) 
            upper = max(right_max, A[i] + K)    
            result = min(upper - lower, result)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLUSlength(self, a, b):
        if a == b:
            return -1
        return max(len(a), len(b))
EOF
if __name__ == "__main__":
    t = int(input().strip())
    pattern = '^[+-]?[0-9]*\.[0-9]+$'
    for _ in range(t):
        print(bool(re.match(pattern, input())))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canCross(self, stones):
        jumps = {}  
        for stone in stones:
            jumps[stone] = set()
        jumps[0].add(0)
        for stone in stones:
            for jump in jumps[stone]:
                for shift in [-1, 0, 1]:
                    if jump + shift > 0 and stone + jump + shift in jumps:
                        jumps[stone + jump + shift].add(jump + shift)
        return bool(jumps[stones[-1]])
EOF
class Solution(object):
    def ways(self, pizza, k):
        MOD = 10**9+7
        prefix = [[0]*len(pizza[0]) for _ in xrange(len(pizza))]
        for j in reversed(xrange(len(pizza[0]))):
            accu = 0
            for i in reversed(xrange(len(pizza))):
                accu += int(pizza[i][j] == 'A')
                prefix[i][j] = (prefix[i][j+1] if (j+1 < len(pizza[0])) else 0) + accu
        dp = [[[0]*k for _ in xrange(len(pizza[0]))] for _ in xrange(len(pizza))]
        for i in reversed(xrange(len(pizza))):
            for j in reversed(xrange(len(pizza[0]))):
                dp[i][j][0] = 1
                for m in xrange(1, k):
                    for n in xrange(i+1, len(pizza)):
                        if prefix[i][j] == prefix[n][j]:
                            continue
                        if prefix[n][j] == 0:
                            break
                        dp[i][j][m] = (dp[i][j][m] + dp[n][j][m-1]) % MOD
                    for n in xrange(j+1, len(pizza[0])):
                        if prefix[i][j] == prefix[i][n]:
                            continue
                        if prefix[i][n] == 0:
                            break
                        dp[i][j][m] = (dp[i][j][m] + dp[i][n][m-1]) % MOD
        return dp[0][0][k-1]
EOF
def anagram(s):
    if len(s)%2 == 1:
        return -1
    res = 0
    cnt1 = Counter(s[:len(s)//2])
    cnt2 = Counter(s[len(s)//2:])
    cnt3 = {}
    for let, val in cnt1.items():
        cnt3[let] = abs(val - cnt2[let])
    for let, val in cnt2.items():
        cnt3[let] = abs(val - cnt1[let])
    for el in cnt3.values():
        res += el
    return res//2
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = anagram(s)
    print(result)
EOF
def rotate(A, pos):
    A[pos], A[pos+1], A[pos+2] = A[pos+1], A[pos+2], A[pos]
def larrysArray(A):
    for _ in range(len(A)):
        for ind in range(1, len(A) - 1):
            a, b, c = A[ind-1], A[ind], A[ind+1]
            if a > b or c < a:
                rotate(A, ind-1)
    if A == sorted(A):
        return 'YES'
    else:
        return 'NO'
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = int(input().strip())
        A = list(map(int, input().strip().split(' ')))
        result = larrysArray(A)
        print(result)
EOF
if __name__ == "__main__":
    T = int(input().strip())
    correct = 1 << 32
    for _ in range(T):
        num = int(input().strip())
        print(~num + correct)
EOF
def gameOfThrones(s):
    cnt = Counter(s)
    if len(s)%2 == 0:
        ret = all([x%2 == 0 for x in cnt.values()])
    else:
        if len(list(filter(lambda x: x%2 == 1, cnt.values()))) == 1:
            ret = True
        else:
            ret = False
    return 'YES' if ret else 'NO'
s = input().strip()
result = gameOfThrones(s)
print(result)
EOF
class Solution(object):
    def numTrees(self, n):
        if n == 0:
            return 1
        def combination(n, k):
            count = 1
            for i in xrange(1, k + 1):
                count = count * (n - i + 1) / i
            return count
        return combination(2 * n, n) - combination(2 * n, n - 1)
class Solution2(object):
    def numTrees(self, n):
        counts = [1, 1]
        for i in xrange(2, n + 1):
            count = 0
            for j in xrange(i):
                count += counts[j] * counts[i - j - 1]
            counts.append(count)
        return counts[-1]
EOF
class Solution(object):
    def leastInterval(self, tasks, n):
        counter = Counter(tasks)
        _, max_count = counter.most_common(1)[0]
        result = (max_count-1) * (n+1)
        for count in counter.values():
            if count == max_count:
                result += 1
        return max(result, len(tasks))
EOF
def lcs(X , Y): 
    m = len(X) 
    n = len(Y) 
    L = [[None]*(n+1) for i in range(m+1)] 
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
    return L[m][n]
def commonChild(s1, s2):
    common_letters = set(s1) & set(s2)
    print("intersect: {}".format(common_letters))
    if (not bool(common_letters)):
        return 0
    s1_filt = "".join([x for x in s1 if x in common_letters])
    s2_filt = "".join([x for x in s2 if x in common_letters])
    print("s1_filt: {}".format(s1_filt))
    print("s2_filt: {}".format(s2_filt))
    return lcs(s1, s2)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s1 = input()
    s2 = input()
    result = commonChild(s1, s2)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
class Solution(object):
    def findBlackPixel(self, picture, N):
        rows, cols = [0] * len(picture),  [0] * len(picture[0])
        lookup = collections.defaultdict(int)
        for i in xrange(len(picture)):
            for j in xrange(len(picture[0])):
                if picture[i][j] == 'B':
                    rows[i] += 1
                    cols[j] += 1
            lookup[tuple(picture[i])] += 1
        result = 0
        for i in xrange(len(picture)):
            if rows[i] == N and lookup[tuple(picture[i])] == N:
                for j in xrange(len(picture[0])):
                     result += picture[i][j] == 'B' and cols[j] == N
        return result
class Solution2(object):
    def findBlackPixel(self, picture, N):
        lookup = collections.Counter(map(tuple, picture))
        cols = [col.count('B') for col in zip(*picture)]
        return sum(N * zip(row, cols).count(('B', N)) \
                   for row, cnt in lookup.iteritems() \
                   if cnt == N == row.count('B'))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def kEmptySlots(self, flowers, k):
        n = len(flowers)
        days = [None for _ in range(n)]         
        for day, pos in enumerate(flowers, 1):  
            days[pos - 1] = day                 
        left, right = 0, k + 1                  
        first_day = n + 1                       
        while right < n:
            for i in range(left + 1, right):    
                if days[i] < days[left] or days[i] < days[right]:   
                    left, right = i, i + k + 1  
                    break
            else:
                first_day = min(first_day, max(days[left], days[right]))    
                left, right = right, right + k + 1  
        return -1 if first_day == n + 1 else first_day
EOF
n, m = [int(x) for x in input().strip().split()]
arr = []
for _ in range(n):
    arr.append([int(x) for x in input().strip().split()])
print(numpy.array(arr).transpose())
print(numpy.array(arr).flatten())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def inorderTraversal(self, root):
        stack, result = [], []
        while root:         
            stack.append(root)
            root = root.left
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
        return result
EOF
def jumpingOnClouds(c):
    jump = -1
    i = 0
    while i < len(c):
        if i < len(c) - 2 and c[i+2] == 0:
            i += 1
        jump += 1
        i += 1
    return jump
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    c = list(map(int, input().rstrip().split()))
    result = jumpingOnClouds(c)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countSegments(self, s):
        return len(s.split())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestLine(self, M):
        if not M or not M[0]:
            return 0
        rows, cols = len(M), len(M[0])
        max_len = 0
        previous_dp = [[0 for _ in range(4)] for c in range(cols)]
        for r in range(rows):
            row_dp = []
            for c in range(cols):
                if M[r][c] == 0:  
                    row_dp.append([0 for _ in range(4)])
                    continue
                row_dp.append([1 for _ in range(4)])  
                if c != 0:
                    row_dp[-1][0] += row_dp[-2][0]  
                row_dp[-1][1] += previous_dp[c][1]  
                if c != 0:                          
                    row_dp[-1][2] += previous_dp[c - 1][2]
                if c != cols - 1:                   
                    row_dp[-1][3] += previous_dp[c + 1][3]
                max_len = max(max_len, max(row_dp[-1]))
            previous_dp = row_dp
        return max_len
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minAreaRect(self, points):
        rows, cols = set(), set()               
        for r, c in points:
            rows.add(r)
            cols.add(c)
        row_to_cols = defaultdict(list)         
        if len(rows) > len(cols):
            for r, c in points:
                row_to_cols[r].append(c)
        else:                                   
            for r, c in points:
                row_to_cols[c].append(r)
        result = float("inf")
        col_pair_to_row = {}                    
        for r in sorted(row_to_cols):
            columns = sorted(row_to_cols[r])
            for i, c1 in enumerate(columns[:-1]):
                for c2 in columns[i + 1:]:
                    if (c1, c2) in col_pair_to_row:     
                        result = min(result, (r - col_pair_to_row[c1, c2]) * (c2 - c1))
                    col_pair_to_row[c1, c2] = r
        return 0 if result == float('inf') else result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minDistance(self, word1, word2):
        def edit_distance(i, j):
            if i < 0 or j < 0:
                return i + 1 + j + 1
            if (i, j) in memo:
                return memo[(i, j)]
            if word1[i] == word2[j]:
                result = edit_distance(i - 1, j - 1)
            else:
                result = 1 + min(edit_distance(i - 1, j),
                                 edit_distance(i, j - 1),
                                 edit_distance(i - 1, j - 1))
            memo[(i, j)] = result
            return result
        memo = {}
        return edit_distance(len(word1) - 1, len(word2) - 1)
EOF
if __name__ == "__main__":
    n = int(input().strip())
    english = set(map(int, input().strip().split(' ')))
    m = int(input().strip())
    french = set(map(int, input().strip().split(' ')))
    print(len(english.symmetric_difference(french)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def PredictTheWinner(self, nums):
        def helper(left, right):
            if right < left:            
                return 0
            if right == left:           
                return nums[left]
            if (left, right) in memo:
                return memo[(left, right)]
            left_right = helper(left + 1, right - 1)
            left_left = helper(left + 2, right)
            take_left = nums[left] + min(left_right - nums[right], left_left - nums[left + 1])
            right_right = helper(left, right - 2)       
            take_right = nums[right] + min(left_right - nums[left], right_right - nums[right - 1])
            result = max(take_left, take_right)
            memo[(left, right)] = result
            return result
        memo = {}
        return helper(0, len(nums) - 1) >= 0
EOF
if __name__ == "__main__":
    a = int(input().strip())
    b = int(input().strip())
    m = int(input().strip())
    print(pow(a, b))
    print(pow(a, b, m))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestComponentSize(self, A):
        def prime_factors(x):   
            factors = set()
            while x % 2 == 0:   
                factors.add(2)
                x //= 2
            for i in range(3, int(x ** 0.5) + 1, 2):
                while x % i == 0:
                    factors.add(i)
                    x //= i
            if x > 2:           
                factors.add(x)
            return factors
        def find(x):            
            while x != parents[x]:
                parents[x] = parents[parents[x]]    
                x = parents[x]
            return x
        def union(x, y):
            x, y = find(x), find(y)
            if x == y:
                return
            parents[x] = y      
            sizes[y] += sizes[x]
            sizes[x] = 0
        n = len(A)
        parents = [i for i in range(n)] 
        sizes = [1] * n                 
        prime_to_index = {}             
        for i, a in enumerate(A):
            primes = prime_factors(a)
            for p in primes:
                if p in prime_to_index:
                    union(i, prime_to_index[p])
                prime_to_index[p] = i
        return max(sizes)
EOF
class Solution(object):
    def printLinkedListInReverse(self, head):
        def print_nodes(head, count):
            nodes = []
            while head and len(nodes) != count:
                nodes.append(head)
                head = head.getNext()
            for node in reversed(nodes):
                node.printValue()
        count = 0
        curr = head
        while curr:
            curr = curr.getNext()
            count += 1
        bucket_count = int(math.ceil(count**0.5))
        buckets = []
        count = 0
        curr = head
        while curr:
            if count % bucket_count == 0:
                buckets.append(curr)
            curr = curr.getNext()
            count += 1
        for node in reversed(buckets):
            print_nodes(node, bucket_count)
class Solution2(object):
    def printLinkedListInReverse(self, head):
        nodes = []
        while head:
            nodes.append(head)
            head = head.getNext()
        for node in reversed(nodes):
            node.printValue()
class Solution3(object):
    def printLinkedListInReverse(self, head):
        tail = None
        while head != tail:
            curr = head
            while curr.getNext() != tail:
                curr = curr.getNext()
            curr.printValue()
            tail = curr
EOF
def angryChildren(k, arr):
    arr = sorted(arr)
    res = arr[-1]
    for ind in range(len(arr)-k+1):
        res = min(res, arr[ind+k-1] - arr[ind])
    return res
if __name__ == "__main__":
    n = int(input().strip())
    k = int(input().strip())
    arr = []
    arr_i = 0
    for arr_i in range(n):
        arr_t = int(input().strip())
        arr.append(arr_t)
    result = angryChildren(k, arr)
    print(result)
EOF
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        result, left = 0, 0
        lookup = {}
        for right in xrange(len(s)):
            if s[right] in lookup:
                left = max(left, lookup[s[right]]+1)
            lookup[s[right]] = right
            result = max(result, right-left+1)
        return result
EOF
class Solution(object):
    def prisonAfterNDays(self, cells, N):
        N -= max(N-1, 0) // 14 * 14  
        for i in xrange(N):
            cells = [0] + [cells[i-1] ^ cells[i+1] ^ 1 for i in xrange(1, 7)] + [0]
        return cells
class Solution2(object):
    def prisonAfterNDays(self, cells, N):
        cells = tuple(cells)
        lookup = {}
        while N:
            lookup[cells] = N
            N -= 1
            cells = tuple([0] + [cells[i - 1] ^ cells[i + 1] ^ 1 for i in xrange(1, 7)] + [0])
            if cells in lookup:
                assert(lookup[cells] - N in (1, 7, 14))
                N %= lookup[cells] - N
                break
        while N:
            N -= 1
            cells = tuple([0] + [cells[i - 1] ^ cells[i + 1] ^ 1 for i in xrange(1, 7)] + [0])
        return list(cells)
EOF
def check_array(k, arr):
    for el1 in arr:
        test_arr = list(arr)
        test_arr.remove(el1)
        for el2 in test_arr:
            if (el1 + el2) % k == 0:
                return False
    return True
def nonDivisibleSubset_brute(k, arr):
    if check_array(k, arr):
        return len(arr)
    best = 0
    for num in range(1, len(arr)):
        to_remove = list(combinations(arr, num))
        for option in to_remove:
            test_arr = list(arr)
            for el in option:
                test_arr.remove(el)
            if check_array(k, test_arr) == True:
                best = max(len(test_arr), best)
    return best
def nonDivisibleSubset(k, arr):
    resid_cnt = [0] * k
    for el in arr:
        resid_cnt[el%k] += 1
    res = min(1, resid_cnt[0])
    for ind in range(1, (k//2)+1):
        if ind != k - ind:
            res += max(resid_cnt[ind], resid_cnt[k - ind])
    if k % 2 == 0 and resid_cnt[int(k/2)] != 0:
        res += 1
    return res
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    arr = list(map(int, input().strip().split(' ')))
    result = nonDivisibleSubset(k, arr)
    print(result)
EOF
def maximumToys(prices, k):
    prices = sorted(prices)
    res = 0
    for el in prices:
        if k - el >= 0:
            k -= el
            res += 1
        else:
            break
    return res
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    prices = list(map(int, input().strip().split(' ')))
    result = maximumToys(prices, k)
    print(result)
EOF
def workbook(n, k, arr):
    res = 0
    page = 1
    for el in arr:
        pr_cur = 1
        for pr in range(pr_cur, pr_cur + el):
            if pr == page + pr//k - (1 if pr%k == 0 else 0):
                res += 1
        if el%k == 0:
            page += el//k
        else:
            page += 1 + el//k
    return res
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    arr = list(map(int, input().strip().split(' ')))
    result = workbook(n, k, arr)
    print(result)
EOF
def countingValleys(n, s):
    res = 0
    in_valley = 0
    curr = 0
    for step in s:
        if step == 'U':
            curr += 1
        else:
            curr -= 1
        if curr < 0 and in_valley == 0:
            in_valley = 1
        if in_valley == 1 and curr == 0:
            in_valley = 0
            res += 1
    return res
if __name__ == "__main__":
    n = int(input().strip())
    s = input().strip()
    result = countingValleys(n, s)
    print(result)
EOF
class Solution(object):
    def __init__(self):
        def dayOfMonth(M):
            return (28 if (M == 2) else 31-(M-1)%7%2)
        self.__lookup = [0]*12
        for M in xrange(1, len(self.__lookup)):
            self.__lookup[M] += self.__lookup[M-1]+dayOfMonth(M)
    def dayOfYear(self, date):
        Y, M, D = map(int, date.split("-"))
        leap = 1 if M > 2 and (((Y % 4 == 0) and (Y % 100 != 0)) or (Y % 400 == 0)) else 0
        return self.__lookup[M-1]+D+leap
class Solution2(object):
    def dayOfYear(self, date):
        def numberOfDays(Y, M):
            leap = 1 if ((Y % 4 == 0) and (Y % 100 != 0)) or (Y % 400 == 0) else 0
            return (28+leap if (M == 2) else 31-(M-1)%7%2)
        Y, M, result = map(int, date.split("-"))
        for i in xrange(1, M):
            result += numberOfDays(Y, i)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isReflected(self, points):
        y_to_x = defaultdict(set)   
        for x, y in points:
            y_to_x[y].add(x)
        reflection = None           
        for y in y_to_x:
            xs = sorted(list(y_to_x[y]))
            left, right = 0, len(xs) - 1
            if reflection is None:
                reflection = xs[left] + xs[right]   
                left += 1
                right -= 1
            while left <= right:
                if xs[right] + xs[left] != reflection :
                    return False
                left += 1
                right -= 1
        return True
EOF
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data.count('\n') > 0:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)
    def handle_data(self, data):
        if len(data) > 1:
            print(">>> Data")
            print(data)
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()
EOF
def countTriplets(arr, r):
    res = 0
    pairs = defaultdict(int)
    triplets = defaultdict(int)
    for el in arr:
        res += triplets[el]
        triplets[r*el] += pairs[el]
        pairs[r*el] += 1
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nr = input().rstrip().split()
    n = int(nr[0])
    r = int(nr[1])
    arr = list(map(int, input().rstrip().split()))
    ans = countTriplets(arr, r)
    fptr.write(str(ans) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def validIPAddress(self, IP):
        ip_list = IP.split(".")
        if len(ip_list) == 4:
            for group in ip_list:
                n = int(group)
                if n < 0 or n > 255 or len(str(n)) != len(group):   
                    return "Neither"
            return "IPv4"
        ip_list = IP.split(":")
        if len(ip_list) != 8:
            return "Neither"
        for group in ip_list:
            n = int(group, 16)
            if n < 0 or n > int("FFFF", 16) or len(group) > 4 or group[0] == "-":   
                return "Neither"
        return "IPv6"
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def mergeLists(head1, head2):
    dummy = SinglyLinkedListNode(-1)
    cur = dummy
    while head1 and head2:
        if head1.data > head2.data:
            cur.next = head2
            head2 = head2.next
        else:
            cur.next = head1
            head1 = head1.next
        cur = cur.next
    if head1 or head2:
        cur.next = head1 or head2
    return dummy.next
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        llist1_count = int(input())
        llist1 = SinglyLinkedList()
        for _ in range(llist1_count):
            llist1_item = int(input())
            llist1.insert_node(llist1_item)
        llist2_count = int(input())
        llist2 = SinglyLinkedList()
        for _ in range(llist2_count):
            llist2_item = int(input())
            llist2.insert_node(llist2_item)
        llist3 = mergeLists(llist1.head, llist2.head)
        print_singly_linked_list(llist3, ' ', fptr)
        fptr.write('\n')
    fptr.close()
EOF
def pokerNim(k, c):
    res = reduce((lambda x, y: x ^ y), c)
    if res == 0:
        return 'Second'
    else:
        return 'First'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        nk = input().split()
        n = int(nk[0])
        k = int(nk[1])
        c = list(map(int, input().rstrip().split()))
        result = pokerNim(k, c)
        fptr.write(result + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class UndirectedGraphNode(object):
    def __init__(self, x):
        self.label = x
        self.neighbors = []
class Solution(object):
    def cloneGraph(self, node):
        if not node:
            return
        cloned_start = UndirectedGraphNode(node.label)
        to_clone = [node]                       
        node_mapping = {node : cloned_start}    
        while to_clone:
            node = to_clone.pop()               
            clone_node = node_mapping[node]
            for neighbor in node.neighbors:
                if neighbor not in node_mapping:    
                    clone_neightbor = UndirectedGraphNode(neighbor.label)
                    node_mapping[neighbor] = clone_neightbor
                    to_clone.append(neighbor)
                else:
                    clone_neightbor = node_mapping[neighbor]
                clone_node.neighbors.append(clone_neightbor)    
        return cloned_start
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minAddToMakeValid(self, S):
        additions = 0               
        net_open = 0                
        for c in S:
            net_open += 1 if c == "(" else -1   
            if net_open == -1:      
                additions += 1
                net_open = 0
        return additions + net_open 
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def isRationalEqual(self, S, T):
        def to_numeric(s):
            if not ("(") in s:                      
                return Fraction(s)
            non_repeat, repeat = s.split("(")
            repeat = repeat[:-1]                    
            _, non_repeat_decimal = non_repeat.split(".")
            fract = Fraction(int(repeat), (10 ** len(repeat) - 1) * (10 ** len(non_repeat_decimal)))
            return Fraction(non_repeat) + fract
        return to_numeric(S) == to_numeric(T)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        if denominator == 0:
            return None
        decimal = []                        
        if numerator * denominator < 0:     
            decimal.append('-')
        output, remainder = divmod(abs(numerator), abs(denominator))    
        decimal.append(str(output))
        if remainder == 0:
            return "".join(decimal)
        decimal.append('.')
        seen = {}                           
        while remainder != 0:
            if remainder in seen:
                return "".join(decimal[:seen[remainder]] + ['('] + decimal[seen[remainder]:] + [')'])
            seen[remainder] = len(decimal)
            output, remainder = divmod(remainder*10, abs(denominator))
            decimal.append(str(output))
        return "".join(decimal)
EOF
def solve(n):
    res = 1
    n_bin = bin(n).replace('0b', '')
    if (n == 0):
        return 1
    for digit in n_bin:
        if digit == '0':
            res *= 2
    return res
n = int(input().strip())
result = solve(n)
print(result)
EOF
def bonAppetit(n, skip, b, ar):
    sum_actual = 0
    ar.pop(skip)
    sum_actual = sum(ar)//2
    if sum_actual == b:
        return 'Bon Appetit'
    else:
        return b - sum_actual
n, k = input().strip().split(' ')
n, k = [int(n), int(k)]
ar = list(map(int, input().strip().split(' ')))
b = int(input().strip())
result = bonAppetit(n, k, b, ar)
print(result)
EOF
class Solution(object):
    def numberOfBoomerangs(self, points):
        result = 0
        for i in xrange(len(points)):
            group = collections.defaultdict(int)
            for j in xrange(len(points)):
                if j == i:
                    continue
                dx, dy =  points[i][0] - points[j][0], points[i][1] - points[j][1]
                group[dx**2 + dy**2] += 1
            for _, v in group.iteritems():
                if v > 1:
                    result += v * (v-1)
        return result
    def numberOfBoomerangs2(self, points):
        cnt = 0
        for a, i in enumerate(points):
            dis_list = []
            for b, k in enumerate(points[:a] + points[a + 1:]):
                dis_list.append((k[0] - i[0]) ** 2 + (k[1] - i[1]) ** 2)
            for z in collections.Counter(dis_list).values():
                if z > 1:
                    cnt += z * (z - 1)
        return cnt
EOF
class Node:
    def __init__(self):
        self.left = None
        self.right = None
    def is_leaf(self):
        return self.left is None and self.right is None
class Trie:
    def __init__(self, size):
        self.root = Node()
        self.size = size
    def add(self, number):
        node = self.root
        for digit in number:
            if digit == '1':
                if node.right is not None:
                    node = node.right
                else:
                    node.right = Node()
                    node = node.right
            if digit == '0':
                if node.left is not None:
                    node = node.left
                else:
                    node.left = Node()
                    node = node.left
    def find_xor_max(self, number):
        res = ''
        untouched = ''
        number_bin = dec_to_bin(number)
        to_maximize = number_bin
        if len(number_bin) > self.size:
            to_maximize = number_bin[len(number_bin) - self.size:]
            untouched = number_bin[:len(number_bin) - self.size]
        elif len(number_bin) < self.size:
            to_maximize = buildup_to_len(number_bin, self.size)
        node = self.root
        for digit in to_maximize:
            if digit == '1':
                if node.left is not None:
                    res += '0'
                    node = node.left
                elif node.right is not None:
                    res += '1'
                    node = node.right
            elif digit == '0':
                if node.right is not None:
                    res += '1'
                    node = node.right
                elif node.left is not None:
                    res += '0'
                    node = node.left
        return bin_to_dec(res) ^ number
    def print_trie(self):
        if self.root != None:
            self._print_trie(self.root, self.size, 0, '')
    def _print_trie(self, node, size, depth, result):
        if node != None:
            if node.is_leaf() and size == depth:
                print(result)
                return
            self._print_trie(node.left, size, depth+1, result + '0')
            self._print_trie(node.right, size, depth+1, result + '1')
def dec_to_bin(number):
    return str(bin(number).replace('0b', ''))
def bin_to_dec(number):
    return int(number, 2)
def make_arr_bin(arr):
    arr_bin = [ dec_to_bin(el) for el in arr ]
    return arr_bin
def get_max_len(arr):
    return len(dec_to_bin(max(arr)))
def buildup_to_len(number, length):
    return '0' * (length - len(number)) + number
def build_trie(arr):
    max_number_len = get_max_len(arr)
    arr_bin = make_arr_bin(arr)
    trie = Trie(max_number_len)
    for elem in arr_bin:
        trie.add(buildup_to_len(elem, max_number_len))
    return trie
if __name__ == "__main__":
    p = int(input().strip())
    q = int(input().strip())
    trie = build_trie(range(p, q+1))
    res_max = 0
    for a in range(p, q + 1):
        get_max = trie.find_xor_max(a)
        if get_max > res_max:
            res_max = get_max
    print(res_max)
EOF
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(0)
        current, carry = dummy, 0
        while l1 or l2:
            val = carry
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            carry, val = divmod(val, 10)
            current.next = ListNode(val)
            current = current.next
        if carry == 1:
            current.next = ListNode(1)
        return dummy.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def combinationSum4(self, nums, target):
        memo = {}
        self.helper(nums, target, memo)
        return memo[target]
    def helper(self, nums, target, memo):
        if target < 0:
            return 0
        if target == 0:
            return 1
        if target in memo:
            return memo[target]
        combos = 0
        for num in nums:
            combos += self.helper(nums, target - num, memo)
        memo[target] = combos
        return combos
class Solution2(object):
    def combinationSum4(self, nums, target):
        combos = [0] * (target + 1)  
        combos[0] = 1
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    combos[i] += combos[i - num]
        return combos[-1]
EOF
class Solution(object):
    def countTriplets(self, A):
        def FWT(A, v):
            B = A[:]
            d = 1
            while d < len(B):
                for i in xrange(0, len(B), d << 1):
                    for j in xrange(d):
                        B[i+j] += B[i+j+d] * v
                d <<= 1
            return B
        k = 3
        n, max_A = 1, max(A)
        while n <= max_A:
            n *= 2
        count = collections.Counter(A)
        B = [count[i] for i in xrange(n)]
        C = FWT(map(lambda x : x**k, FWT(B, 1)), -1)
        return C[0]
class Solution2(object):
    def countTriplets(self, A):
        count = collections.defaultdict(int)
        for i in xrange(len(A)):
            for j in xrange(len(A)):
                count[A[i]&A[j]] += 1
        result = 0
        for k in xrange(len(A)):
            for v in count:
                if A[k]&v == 0:
                    result += count[v]
        return result
EOF
def freqQuery(queries):
    database = defaultdict(int)
    frequences = defaultdict(int)
    output = list()
    for q in queries:
        op, val = q
        if op == 1:
            frequences[database[val]] = max(0, frequences[database[val]]-1)
            database[val] += 1
            frequences[database[val]] += 1
        elif op == 2:
            frequences[database[val]] = max(0, frequences[database[val]]-1)
            database[val] = max(0, database[val] - 1)
            frequences[database[val]] += 1
        elif op == 3:
            if frequences[val] > 0:
                output.append(1)
            else:
                output.append(0)
    return output
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input().strip())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().rstrip().split())))
    ans = freqQuery(queries)
    fptr.write('\n'.join(map(str, ans)))
    fptr.write('\n')
    fptr.close()
EOF
def rotLeft(nums, k):
    k = k % len(nums)
    count = start = len(nums)-1
    while count >=0 :
        cur = start
        prev = nums[start]
        while 1:
            nextt = (cur - k) % len(nums)
            nums[nextt], prev = prev, nums[nextt]
            cur  = nextt
            count -= 1
            if start == cur:
                break
        start -= 1
    return a
if __name__ == '__main__':
    nd = input().split()
    n = int(nd[0])
    d = int(nd[1])
    a = list(map(int, input().rstrip().split()))
    print(*rotLeft(a,d))
EOF
class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class Solution:
    def display(self,head):
        current = head
        while current:
            print(current.data,end=' ')
            current = current.next
    def insert(self, head, data):
        current = head
        new_node = Node(data)
        new_node.next = head
        cuurent = new_node
        return new_node
mylist = Solution()
T = int(input())
head = None
for i in range(T):
    data = int(input())
    head = mylist.insert(head, data)
mylist.display(head);
EOF
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def search(self, nums, target):
        left, right = 0, len(nums) - 1      
        while left <= right:
            mid = (left + right) // 2       
            if target == nums[mid]:
                return mid
            if target > nums[mid]:          
                left = mid + 1
            else:                           
                right = mid - 1
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def myPow(self, x, n):
        neg = n < 0
        pos_result = self.pos_pow(x, abs(n))
        return 1/pos_result if neg else pos_result
    def pos_pow(self, x, n):
        if n == 0:
            return 1
        temp = self.pos_pow(x, n//2)
        temp *= temp
        return temp if n % 2 == 0 else temp * x
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findRestaurant(self, list1, list2):
        if len(list1) > len(list2):         
            list1, list2 = list2, list1
        dict1 = {rest: i for i, rest in enumerate(list1)}
        result = []
        min_sum = float("inf")
        for i, rest in enumerate(list2):
            if i > min_sum:                 
                break
            if rest not in dict1:
                continue
            sum_i = i + dict1[rest]
            if sum_i < min_sum:             
                min_sum = sum_i
                result = [rest]
            elif sum_i == min_sum:          
                result.append(rest)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def diffWaysToCompute(self, input):
        start = 0       
        parsed = []
        for i in range(len(input)):
            if not input[i].isdigit():
                parsed.append(int(input[start:i]))      
                parsed.append(input[i])                 
                start = i+1
        parsed.append(int(input[start:len(input)]))
        return self.diff_ways(parsed, 0, len(parsed)-1, {})
    def diff_ways(self, s, left, right, memo):
        if left == right:       
            return [s[left]]
        if (left, right) in memo:
            return memo[(left, right)]
        ways = []
        for i in range(left+1, right, 2):   
            left_results = self.diff_ways(s, left, i-1, memo)
            right_results = self.diff_ways(s, i+1, right, memo)
            for l in left_results:
                for r in right_results:
                    if s[i] == '+':
                        ways.append(l+r)
                    elif s[i] == '-':
                        ways.append(l-r)
                    elif s[i] == '*':
                        ways.append(l*r)
        memo[(left, right)] = ways
        return ways
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        if len(s1) + len(s2) != len(s3):    
            return False
        return self.helper(s1, s2, s3, 0, 0, {})
    def helper(self, s1, s2, s3, i, j, memo):   
        if i >= len(s1) or j >= len(s2):        
            return s1[i:] + s2[j:] == s3[i+j:]
        if (i, j) in memo:
            return memo[(i, j)]
        result = False
        if s1[i] == s3[i+j] and self.helper(s1, s2, s3, i+1, j, memo):
            result = True
        elif s2[j] == s3[i+j] and self.helper(s1, s2, s3, i, j+1, memo):
            result = True
        memo[(i, j)] = result
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def generateParenthesis(self, n):
        result = []
        self.generate([], n, n, result)
        return result
    def generate(self, prefix, left, right, result):
        if left == 0 and right == 0:
            result.append("".join(prefix))
        if left != 0:
            self.generate(prefix + ['('], left-1, right, result)
        if right > left:
            self.generate(prefix + [')'], left, right-1, result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, nums):
        self.nums = nums
    def reset(self):
        return self.nums
    def shuffle(self):
        result = self.nums[:]
        for i in range(len(result)):
            swap = random.randint(i, len(result) - 1)
            result[i], result[swap] = result[swap], result[i]
        return result
EOF
n = int(input().strip())
matrix = []
for i in range(n):
    matrix.append(list(map(int, input().split())))
sum1 = sum2 = 0
for i in range(len(matrix)):
    sum1 += matrix[i][i]
    sum2 += matrix[i][n-1-i]
print(abs(sum1-sum2))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def titleToNumber(self, s):
        result = 0
        for c in s:
            result = result*26 + ord(c) - ord('A') + 1
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def pushDominoes(self, dominos):
        prev_R = float("-inf")              
        rights = []                         
        for i, c in enumerate(dominos):
            rights.append(prev_R)
            if c == "R":                    
                prev_R = i
            elif c == "L":                  
                prev_R = float("-inf")
        prev_L = float("inf")               
        lefts = [0] * len(dominos)          
        for i in range(len(dominos) - 1, -1, -1):
            lefts[i] = prev_L
            if dominos[i] == "L":           
                prev_L = i
            elif dominos[i] == "R":         
                prev_L = float("inf")
        dominos = [c for c in dominos]
        for i in range(len(dominos)):
            if dominos[i] == ".":           
                diff = (lefts[i] - i) - (i - rights[i]) 
                if diff < 0:
                    dominos[i] = "L"
                elif diff > 0:
                    dominos[i] = "R"
        return "".join(dominos)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getMaxRepetitions(self, s1, n1, s2, n2):
        if any(c for c in set(s2) if c not in set(s1)):     
            return 0
        i, j = 0, 0                                         
        s1_reps, s2_reps = 0, 0                             
        s2_index_to_reps = {0 : (0, 0)}                     
        while s1_reps < n1:                                 
            if s1[i] == s2[j]:
                j += 1                                      
            i += 1
            if j == len(s2):
                j = 0
                s2_reps += 1
            if i == len(s1):
                i = 0
                s1_reps += 1
                if j in s2_index_to_reps:                   
                    break
                s2_index_to_reps[j] = (s1_reps, s2_reps)    
        if s1_reps == n1:                                   
            return s2_reps // n2
        initial_s1_reps, initial_s2_reps = s2_index_to_reps[j]
        loop_s1_reps = s1_reps - initial_s1_reps
        loop_s2_reps = s2_reps - initial_s2_reps
        loops = (n1 - initial_s1_reps) // loop_s1_reps      
        s1_reps = initial_s1_reps + loops * loop_s1_reps
        s2_reps = initial_s2_reps + loops * loop_s2_reps
        while s1_reps < n1:                                 
            if s1[i] == s2[j]:
                j += 1
            i += 1
            if i == len(s1):
                i = 0
                s1_reps += 1
            if j == len(s2):
                j = 0
                s2_reps += 1
        return s2_reps // n2
EOF
print(len(set(input() for _ in range(int(input())))))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MovingAverage(object):
    def __init__(self, size):
        self.array = [None for _ in range(size)]    
        self.i = 0                                  
        self.total = 0                              
    def next(self, val):
        if self.array[self.i] is not None:          
            self.total -= self.array[self.i]
        self.total += val
        self.array[self.i] = val
        self.i = (self.i + 1) % len(self.array)
        count = len(self.array)                     
        if self.array[-1] is None:
            count = self.i
        return self.total / float(count)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def scoreOfParentheses(self, S):
        stack = []                  
        for s in S:
            if s == "(":
                stack.append(s)
            else:                   
                item = stack.pop()
                if item == "(":     
                    num = 1
                else:               
                    stack.pop()     
                    num = 2 * item
                if stack and stack[-1] != "(":  
                    stack[-1] += num
                else:
                    stack.append(num)
        return stack[0]
EOF
if __name__ == '__main__':
    n = int(input())
    print(len(max(bin(n)[2:].split("0"))))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def removeDuplicateLetters(self, s):
        s_set = sorted(set(s))
        for c in s_set:
            suffix = s[s.index(c):]
            if len(set(suffix)) == len(s_set):
                return c + self.removeDuplicateLetters(suffix.replace(c, ""))
        return ""
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def removeInvalidParentheses(self, s):
        valid = []
        self.remove(s, valid, 0, 0, ('(', ')'))
        return valid
    def remove(self, s, valid, start, removed,  par):
        net_open = 0
        for i in range(start, len(s)):
            net_open += ((s[i] == par[0]) - (s[i] == par[1]))
            if net_open >= 0:
                continue
            for j in range(removed, i+1):
                if s[j] == par[1] and (j == removed or s[j - 1] != par[1]):
                    self.remove(s[:j] + s[j+1:], valid, i, j, par)
            return
        reversed = s[::-1]
        if par[0] == '(':       
            self.remove(reversed, valid, 0, 0, (')', '('))
        else:                   
            valid.append(reversed)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minMalwareSpread(self, graph, initial):
        neighbours = {}                     
        for i, row in enumerate(graph):
            neighbours[i] = [j for j, val in enumerate(row) if val == 1 and j != i]
        def infected():                     
            for node in initial:
                connected(node)
            return len(visited)
        def connected(node):                
            if node in visited:
                return
            visited.add(node)
            for nbor in neighbours[node]:
                connected(nbor)
        visited = set()
        initial_infected = infected()
        best_gain = 0                       
        best_node = None
        for removed in sorted(initial):     
            visited = {removed}             
            infected()
            gain = initial_infected - len(visited) + 1  
            if gain > best_gain:
                best_gain = gain
                best_node = removed
        return best_node
EOF
class Calculator:
    def power(self,n,p):
        if n < 0 or p < 0:
            raise Exception('n and p should be non-negative')
        return n**p
myCalculator=Calculator()
T=int(input())
for i in range(T):
    n,p = map(int, input().split())
    try:
        ans=myCalculator.power(n,p)
        print(ans)
    except Exception as e:
        print(e)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def minMeetingRooms(self, intervals):
        max_rooms = 0
        rooms = []                              
        intervals.sort(key=lambda x: x.start)  
        for interval in intervals:
            heapq.heappush(rooms, interval.end)
            while rooms[0] <= interval.start:   
                heapq.heappop(rooms)
            max_rooms = max(max_rooms, len(rooms))
        return max_rooms
class Solution2(object):
    def minMeetingRooms(self, intervals):
        overlaps = []
        intervals.sort(key=lambda x: x.start)
        for interval in intervals:
            if overlaps and interval.start >= overlaps[0]:  
                heapq.heapreplace(overlaps, interval.end)
            else:                                           
                heapq.heappush(overlaps, interval.end)
        return len(overlaps)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def largestPerimeter(self, A):
        A.sort(reverse=True)
        for i, side in enumerate(A[:-2]):
            if side < A[i + 1] + A[i + 2]:
                return side + A[i + 1] + A[i + 2]
        return 0
EOF
for _ in range(int(input())):
    try:
        re.compile(input())
        print('True')
    except re.error:
        print("False")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def loudAndRich(self, richer, quiet):
        n = len(quiet)
        richer_than = [set() for _ in range(n)]     
        for a, b in richer:
            richer_than[b].add(a)
        result = [None] * n
        def update_results(person):
            if result[person] is not None:          
                return
            result[person] = person                 
            for rich in richer_than[person]:
                update_results(rich)                
                if quiet[result[rich]] < quiet[result[person]]: 
                    result[person] = result[rich]   
        for i in range(n):
            update_results(i)
        return result
EOF
def ransom_note(magazine, ransom):
    magazine_dict = {}
    for i in magazine:
        if i in magazine_dict:
            magazine_dict[i] +=  1
        else:
            magazine_dict[i] = 1
    for i in ransom:
        if i in magazine_dict:
            if magazine_dict[i] == 0:
                return False
            else:
                magazine_dict[i] -= 1
        else:
            return False
    return True
m, n = map(int, input().strip().split(' '))
magazine = list(input().strip().split(' '))
ransom = list(input().split(' '))
answer = ransom_note(magazine, ransom)
if (answer):
    print("Yes")
else:
    print("No")
EOF
arr = np.array(input().split(), int)
print(np.reshape(arr,(3,3)))
EOF
class Solution(object):
    def eventualSafeNodes(self, graph):
        WHITE, GRAY, BLACK = 0, 1, 2
        def dfs(graph, node, lookup):
            if lookup[node] != WHITE:
                return lookup[node] == BLACK
            lookup[node] = GRAY
            for child in graph[node]:
                if lookup[child] == BLACK:
                    continue
                if lookup[child] == GRAY or \
                   not dfs(graph, child, lookup):
                    return False
            lookup[node] = BLACK
            return True
        lookup = collections.defaultdict(int)
        return filter(lambda node: dfs(graph, node, lookup), xrange(len(graph)))
EOF
def stones(n, a, b):
    return sorted(set([(n-1)*min(a, b) + x*abs(a-b) for x in range(n)]))
if __name__ == "__main__":
    T = int(input().strip())
    for a0 in range(T):
        n = int(input().strip())
        a = int(input().strip())
        b = int(input().strip())
        result = stones(n, a, b)
        print (" ".join(map(str, result)))
EOF
def saveThePrisoner(n, m, s):
    res = (s + m-1) % n
    return res if res != 0 else n
t = int(input().strip())
for a0 in range(t):
    n, m, s = input().strip().split(' ')
    n, m, s = [int(n), int(m), int(s)]
    result = saveThePrisoner(n, m, s)
    print(result)
EOF
def Reverse(head):
    prev = None
    node = head
    while node is not None:
        buf = node.next
        node.next = prev
        prev = node
        node = buf
    head = prev
    return head
EOF
class Solution(object):
    def prefixesDivBy5(self, A):
        for i in xrange(1, len(A)):
            A[i] += A[i-1] * 2 % 5
        return [x % 5 == 0 for x in A]
EOF
def surfaceArea(A):
    H, W = len(A), len(A[0])
    area = 2*H*W
    for ind in range(H):
        for jnd in range(W):
            if ind-1 >= 0:
                area += max(0, A[ind][jnd] - A[ind-1][jnd])
            else:
                area += A[ind][jnd]
            if jnd-1 >= 0:
                area += max(0, A[ind][jnd] - A[ind][jnd-1])
            else:
                area += A[ind][jnd]
            if ind+1 < H:
                area += max(0, A[ind][jnd] - A[ind+1][jnd])
            else:
                area += A[ind][jnd]
            if jnd+1 < W:
                area += max(0, A[ind][jnd] - A[ind][jnd+1])
            else:
                area += A[ind][jnd]
    return area
if __name__ == "__main__":
    H, W = input().strip().split(' ')
    H, W = [int(H), int(W)]
    A = []
    for A_i in range(H):
        A_t = [int(A_temp) for A_temp in input().strip().split(' ')]
        A.append(A_t)
    result = surfaceArea(A)
    print(result)
EOF
class Solution(object):
    def findPeakElement(self, nums):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) / 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left
EOF
def minimumDistances(array):
    res = -1
    memo = [-1] * (10**5 + 3)
    for ind, el in enumerate(array):
        if memo[el] >= 0:
            res = min(res if res >= 0 else 10**5 + 2, ind - memo[el])
        memo[el] = ind
    return res
if __name__ == "__main__":
    n = int(input().strip())
    a = list(map(int, input().strip().split(' ')))
    result = minimumDistances(a)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, n_rows, n_cols):
        self.start, self.end = 0, n_rows * n_cols - 1
        self.used_to_free = {}      
        self.cols = n_cols
    def flip(self):
        x = random.randint(self.start, self.end)
        index = self.used_to_free.get(x, x)         
        self.used_to_free[x] = self.used_to_free.get(self.start, self.start) 
        self.start += 1
        return list(divmod(index, self.cols))       
    def reset(self):
        self.start = 0
        self.used_to_free = {}
EOF
class Node:
    def __init__(self, info): 
        self.info = info  
        self.left = None  
        self.right = None 
        self.level = None 
    def __str__(self):
        return str(self.info) 
class BinarySearchTree:
    def __init__(self): 
        self.root = None
    def create(self, val):  
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break
def height(root):
    if not root:
        return -1
    return 1 + max(height(root.left),height(root.right))
tree = BinarySearchTree()
t = int(input())
arr = list(map(int, input().split()))
for i in range(t):
    tree.create(arr[i])
print(height(tree.root))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def topKFrequent(self, words, k):
        freq = Counter(words)
        pairs = [(-count, word) for word, count in freq.items()]
        heapq.heapify(pairs)
        return [heapq.heappop(pairs)[1] for _ in range(k)]
EOF
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print([ [ i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if ( (i + j + k) != n )])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def catMouseGame(self, graph):
        DRAW, MOUSE, CAT = 0, 1, 2
        n = len(graph)
        def parents(mouse, cat, turn):      
            if turn == CAT:
                return [(new_mouse, cat, 3 - turn) for new_mouse in graph[mouse]]
            return [(mouse, new_cat, 3 - turn) for new_cat in graph[cat] if new_cat != 0]
        state_winner = defaultdict(int)     
        degree = {}                         
        for mouse in range(n):
            for cat in range(n):
                degree[mouse, cat, MOUSE] = len(graph[mouse])
                degree[mouse, cat, CAT] = len(graph[cat]) - (0 in graph[cat])
        queue = deque()                     
        for i in range(n):
            for turn in [MOUSE, CAT]:
                state_winner[0, i, turn] = MOUSE    
                queue.append((0, i, turn, MOUSE))
                if i > 0:                   
                    state_winner[i, i, turn] = CAT
                    queue.append((i, i, turn, CAT))
        while queue:
            i, j, turn, winner = queue.popleft()    
            for i2, j2, prev_turn in parents(i, j, turn):   
                if state_winner[i2, j2, prev_turn] is DRAW:
                    if prev_turn == winner:         
                        state_winner[i2, j2, prev_turn] = winner
                        queue.append((i2, j2, prev_turn, winner))
                    else:                           
                        degree[i2, j2, prev_turn] -= 1
                        if degree[i2, j2, prev_turn] == 0:  
                            state_winner[i2, j2, prev_turn] = turn
                            queue.append((i2, j2, prev_turn, turn))
        return state_winner[1, 2, MOUSE]
EOF
class Solution(object):
    def maxChunksToSorted(self, arr):
        def compare(i1, i2):
            return arr[i1]-arr[i2] if arr[i1] != arr[i2] else i1-i2
        idxs = [i for i in xrange(len(arr))]
        result, max_i = 0, 0
        for i, v in enumerate(sorted(idxs, cmp=compare)):
            max_i = max(max_i, v)
            if max_i == i:
                result += 1
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def dominantIndex(self, nums):
        first_i = 0     
        second = 0      
        for i, num in enumerate(nums[1:], 1):
            if num >= nums[first_i]:
                first_i, second = i, nums[first_i]      
            elif num > second:
                second = num
        return first_i if nums[first_i] >= 2 * second else -1   
EOF
def number_needed(a, b):
    result = 0
    a_dict = dict.fromkeys(string.ascii_lowercase, 0)
    b_dict = dict.fromkeys(string.ascii_lowercase, 0)
    for a_symb in a:
        a_dict[a_symb] = a_dict[a_symb] + 1
    for b_symb in b:
        b_dict[b_symb] = b_dict[b_symb] + 1
    for key in string.ascii_lowercase:
        result += math.fabs(a_dict[key] - b_dict[key])
    return int(result)
a = input().strip()
b = input().strip()
print(number_needed(a, b))
EOF
class Solution(object):
    def minimumSwap(self, s1, s2):
        x1, y1 = 0, 0
        for i in xrange(len(s1)):
            if s1[i] == s2[i]:
                continue
            x1 += int(s1[i] == 'x')
            y1 += int(s1[i] == 'y')
        if x1%2 !=  y1%2:  
            return -1
        return (x1//2 + y1//2) + (x1%2 + y1%2)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class LFUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.time = 0
        self.map = {}  
        self.freq_time = {}  
        self.priority_queue = []  
        self.update = set()  
    def get(self, key):
        self.time += 1
        if key in self.map:
            freq, _ = self.freq_time[key]
            self.freq_time[key] = (freq + 1, self.time)
            self.update.add(key)
            return self.map[key]
        return -1
    def put(self, key, value):
        if self.capacity <= 0:
            return
        self.time += 1
        if not key in self.map:
            if len(self.map) >= self.capacity:  
                while self.priority_queue and self.priority_queue[0][2] in self.update:
                    _, _, k = heapq.heappop(self.priority_queue)
                    f, t = self.freq_time[k]
                    heapq.heappush(self.priority_queue, (f, t, k))
                    self.update.remove(k)
                _, _, k = heapq.heappop(self.priority_queue)
                self.map.pop(k)
                self.freq_time.pop(k)
            self.freq_time[key] = (0, self.time)
            heapq.heappush(self.priority_queue, (0, self.time, key))
        else:
            freq, _ = self.freq_time[key]
            self.freq_time[key] = (freq + 1, self.time)
            self.update.add(key)
        self.map[key] = value
EOF
class Solution(object):
    def checkEqualTree(self, root):
        def getSumHelper(node, lookup):
            if not node:
                return 0
            total = node.val + \
                    getSumHelper(node.left, lookup) + \
                    getSumHelper(node.right, lookup)
            lookup[total] += 1
            return total
        lookup = collections.defaultdict(int)
        total = getSumHelper(root, lookup)
        if total == 0:
            return lookup[total] > 1
        return total%2 == 0 and (total/2) in lookup
EOF
def birthdayCakeCandles(n, ar):
    return ar.count(max(ar))
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = birthdayCakeCandles(n, ar)
print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isPowerOfThree(self, n):
        if n <= 0:
            return False
        max_int = 2 ** 31 - 1
        max_power = int(math.log(max_int, 3))
        return 3 ** max_power % n == 0
EOF
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def isCousins(self, root, x, y):
        def dfs(root, x, depth, parent):
            if not root:
                return False
            if root.val == x:
                return True
            depth[0] += 1
            prev_parent, parent[0] = parent[0], root
            if dfs(root.left, x, depth, parent):
                return True
            parent[0] = root
            if dfs(root.right, x, depth, parent):
                return True
            parent[0] = prev_parent
            depth[0] -= 1
            return False
        depth_x, depth_y = [0], [0]
        parent_x, parent_y = [None], [None]
        return dfs(root, x, depth_x, parent_x) and \
               dfs(root, y, depth_y, parent_y) and \
               depth_x[0] == depth_y[0] and \
               parent_x[0] != parent_y[0]
EOF
        
def is_leaf(root):
    if root.right == None and root.left == None:
        return True
    else:
        return False
def decodeHuff(root , s):
    node = root
    output = ''
    for dig in s:
        if dig == '0':
            if is_leaf(node.left):
                output += node.left.data
                node = root
            else:
                node = node.left
        else:
            if is_leaf(node.right):
                output += node.right.data
                node = root
            else:
                node = node.right
    print output
EOF
def sockMerchant(n, ar):
    socks = {}
    res = 0
    for el in ar:
        if el not in socks.keys():
            socks[el] = 1
        else:
            socks[el] += 1
    for key in socks.keys():
        res += socks[key]//2
    return res
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = sockMerchant(n, ar)
print(result)
EOF
class Cashier(object):
    def __init__(self, n, discount, products, prices):
        self.__n = n
        self.__discount = discount
        self.__curr = 0
        self.__lookup = {p : prices[i] for i, p in enumerate(products)}
    def getBill(self, product, amount):
        self.__curr = (self.__curr+1) % self.__n
        result = 0.0
        for i, p in enumerate(product):
            result += self.__lookup[p]*amount[i]
        return result * (1.0 - self.__discount/100.0 if self.__curr == 0 else 1.0)    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def repeatedNTimes(self, A):
        seen = set()
        for num in A:
            if num in seen:
                return num
            seen.add(num)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def exist(self, board, word):
        if not board or not board[0]:
            return False
        rows, cols = len(board), len(board[0])
        for r in range(rows):
            for c in range(cols):
                if self.can_find(word, 0, board, r, c):
                    return True
        return False
    def can_find(self, word, i, board, r, c):
        if i >= len(word):              
            return True
        if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]):     
            return False
        if word[i] != board[r][c]:      
            return False
        board[r][c] = '*'               
        if (self.can_find(word, i+1, board, r+1, c) or self.can_find(word, i+1, board, r-1, c) or
                self.can_find(word, i+1, board, r, c+1) or self.can_find(word, i+1, board, r, c-1)):
            return True
        board[r][c] = word[i]           
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def regionsBySlashes(self, grid):
        n= len(grid)
        UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
        parents = {}                                
        def find(node):                             
            parents.setdefault(node, node)
            while parents[node] != node:
                parents[node] = parents[parents[node]]  
                node = parents[node]
            return node
        def union(node1, node2):                    
            parent1, parent2 = find(node1), find(node2)
            parents[parent2] = parent1
        for r in range(n):
            for c in range(n):
                if r != n - 1:                   
                    union((r, c, DOWN), (r + 1, c, UP))
                if c != n - 1:                   
                    union((r, c, RIGHT), (r, c + 1, LEFT))
                if grid[r][c] == "/":               
                    union((r, c, UP), (r, c, LEFT))
                    union((r, c, DOWN), (r, c, RIGHT))
                elif grid[r][c] == "\\":
                    union((r, c, UP), (r, c, RIGHT))
                    union((r, c, DOWN), (r, c, LEFT))
                else:                               
                    union((r, c, UP), (r, c, LEFT))
                    union((r, c, UP), (r, c, DOWN))
                    union((r, c, UP), (r, c, RIGHT))
        return len({find(node) for node in parents.keys()})     
EOF
if __name__ == "__main__":
    string = input()
    match = re.search(r'([a-zA-Z0-9])\1+', string)
    print(match.group(1) if match else -1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rob(self, nums):
        if len(nums) < 2:
            return sum(nums)    
        loot, prev = 0, 0
        for num in nums[1:]:    
            loot, prev = max(num + prev, loot), loot
        nums[-1] = 0            
        loot2, prev = 0, 0
        for num in nums:
            loot2, prev = max(num + prev, loot2), loot2
        return max(loot, loot2)
EOF
def plusMinus(arr):
    pos,neg = 0,0
    for i in arr:
        if i > 0:
            pos += 1
        elif i < 0:
            neg += 1
    l = len(arr)
    print("%.6f"%(pos/l))
    print("%.6f"%(neg/l))
    print("%.6f"%((l-(neg+pos))/l))
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    plusMinus(arr)
EOF
def lonely_integer(a):
    res = 0
    for elem in a:
        res ^= elem
    return res
n = int(input().strip())
a = [int(a_temp) for a_temp in input().strip().split(' ')]
print(lonely_integer(a))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def insert(self, head, insertVal):
        if not head:                            
            new_node = Node(insertVal, None)
            new_node.next = new_node
            return new_node
        def insert_after(node):
            node.next = Node(insertVal, node.next)
        original_head = head                    
        while True:
            if head.next.val > head.val:        
                if insertVal >= head.val and insertVal <= head.next.val:    
                    break
            elif head.next.val < head.val:      
                if insertVal >= head.val or insertVal <= head.next.val:     
                    break
            elif head.next == original_head:                                
                break
            head = head.next
        insert_after(head)
        return (original_head)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def diStringMatch(self, S):
        result = []
        low, high = 0, len(S)
        for c in S:
            if c == "I":
                result.append(low)
                low += 1
            else:
                result.append(high)
                high -= 1
        return result + [low]       
EOF
def insertionSort1(n, arr):
    probe = arr[-1]
    for ind in range(len(arr)-2, -1, -1):
        if arr[ind] > probe:
            arr[ind+1] = arr[ind]
            print(" ".join(map(str, arr)))
        else:
            arr[ind+1] = probe
            print(" ".join(map(str, arr)))
            break
    if arr[0] > probe:
        arr[0] = probe
        print(" ".join(map(str, arr)))
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    insertionSort1(n, arr)
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def insertNodeAtHead(llist, data):
    if not llist:
        return SinglyLinkedListNode(data)
    head = SinglyLinkedListNode(data)
    head.next = llist
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist_head = insertNodeAtHead(llist.head, llist_item)
        llist.head = llist_head
    print_singly_linked_list(llist.head, '\n', fptr)
    fptr.write('\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Vector2D(object):
    def __init__(self, vec2d):
        self.vec2d = vec2d
        self.list_nb, self.item_nb = 0, 0
        while self.list_nb < len(self.vec2d) and len(self.vec2d[self.list_nb]) == 0:
            self.list_nb += 1
    def next(self):
        result = self.vec2d[self.list_nb][self.item_nb]
        if self.item_nb < len(self.vec2d[self.list_nb]) - 1:
            self.item_nb += 1   
        else:                   
            self.item_nb = 0
            self.list_nb += 1
            while self.list_nb < len(self.vec2d) and len(self.vec2d[self.list_nb]) == 0:
                self.list_nb += 1
        return result
    def hasNext(self):
        return self.list_nb < len(self.vec2d)   
EOF
if __name__ == "__main__":
    s_num = int(input().strip())
    tuple_fields = input().strip().split()
    student = namedtuple('student', tuple_fields)
    library = []
    res = 0
    for _ in range(s_num):
        st_info = input().strip().split()
        library.append(student(st_info[0], st_info[1], st_info[2], st_info[3]))
    for el in library:
        res += int(el.MARKS)
    print(res/s_num)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class AutocompleteSystem(object):
    def __init__(self, sentences, times):
        self.partial = []  
        self.matches = []  
        self.counts = defaultdict(int)  
        for sentence, count in zip(sentences, times):
            self.counts[sentence] = count
    def input(self, c):
        if c == "
            sentence = "".join(self.partial)
            self.counts[sentence] += 1
            self.partial = []  
            self.matches = []
            return []
        if not self.partial:  
            self.matches = [(-count, sentence) for sentence, count in self.counts.items() if sentence[0] == c]
            self.matches.sort()
            self.matches = [sentence for _, sentence in self.matches]  
        else:
            i = len(self.partial)  
            self.matches = [sentence for sentence in self.matches if len(sentence) > i and sentence[i] == c]
        self.partial.append(c)
        return self.matches[:3]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lastRemaining(self, n):
        head = 1        
        l_to_r = True   
        step = 1        
        while n > 1:    
            if l_to_r:
                head += step
            else:       
                if n % 2 != 0:
                    head += step
            step *= 2
            n //= 2     
            l_to_r = not l_to_r
        return head
EOF
def minimumBribes(q):
    res = 0
    for ind in range(len(q)):
        if q[ind] - (1+ind) > 2:
            return "Too chaotic"
    for ind in range(len(q)):
        for jnd in range(max(0, q[ind]-2), ind):
            if (q[jnd] > q[ind]):
                res += 1
    return res
if __name__ == '__main__':
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        q = list(map(int, input().rstrip().split()))
        print(minimumBribes(q))
EOF
class Solution(object):
    def isPowerOfFour(self, num):
        return num > 0 and (num & (num - 1)) == 0 and \
               ((num & 0b01010101010101010101010101010101) == num)
class Solution2(object):
    def isPowerOfFour(self, num):
        while num and not (num & 0b11):
            num >>= 2
        return (num == 1)
class Solution3(object):
    def isPowerOfFour(self, num):
        num = bin(num)
        return True if num[2:].startswith('1') and len(num[2:]) == num.count('0') and num.count('0') % 2 and '-' not in num else False
EOF
def gemstones(arrays):
    superset = set(arrays[0])
    for arr in arrays[1:]:
        superset &= set(arr)
    return len(superset)
n = int(input().strip())
arr = []
arr_i = 0
for arr_i in range(n):
    arr_t = str(input().strip())
    arr.append(arr_t)
result = gemstones(arr)
print(result)
EOF
class Graph:
    def __init__(self, n):
        self.size = n
        self.vert = dict.fromkeys([n for n in range(n)])
        for node in self.vert.keys():
            self.vert[node] = []
    def print_graph(self):
        print(self.vert)
    def connect(self, x, y):
        self.vert[x].append(y)
        self.vert[y].append(x)
    def find_shortest(self, start):
        next_to_visit = queue.Queue()
        visited = []
        node = start
        next_to_visit.put(node)
        distances = [-1] * self.size
        distances[node] = 0
        while not next_to_visit.empty():
            node = next_to_visit.get()
            height = distances[node]
            for adj in self.vert[node]:
                if not adj in visited:
                    distances[adj] = height + 6
                    next_to_visit.put(adj)
                    visited.append(adj)
        return distances
    def find_all_distances(self, start):
        result = self.find_shortest(start)
        for ind, node in enumerate(result):
            if ind != start:
                print("{} ".format(node), end='')
        print()
t = int(input())
for i in range(t):
    n,m = [int(value) for value in input().split()]
    graph = Graph(n)
    for i in range(m):
        x,y = [int(x) for x in input().split()]
        graph.connect(x-1,y-1) 
    s = int(input())
    graph.find_all_distances(s-1)
EOF
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class CBTInserter(object):
    def __init__(self, root):
        self.__tree = [root]
        for i in self.__tree:
            if i.left:
                self.__tree.append(i.left)
            if i.right:
                self.__tree.append(i.right)        
    def insert(self, v):
        n = len(self.__tree)
        self.__tree.append(TreeNode(v))
        if n % 2:
            self.__tree[(n-1)//2].left = self.__tree[-1]
        else:
            self.__tree[(n-1)//2].right = self.__tree[-1]
        return self.__tree[(n-1)//2].val
    def get_root(self):
        return self.__tree[0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rotate(self, matrix):
        n = len(matrix)
        layers = n//2
        for layer in range(layers):
            for i in range(layer, n - layer - 1):
                temp = matrix[layer][i]
                matrix[layer][i] = matrix[n - 1 - i][layer]
                matrix[n - 1 - i][layer] = matrix[n - 1 - layer][n - 1- i]
                matrix[n - 1 - layer][n - 1 - i] = matrix[i][n - 1 - layer]
                matrix[i][n - 1 - layer] = temp
EOF
def beautifulPairs(arr, brr):
    res = 0
    arr = sorted(arr)
    brr = sorted(brr)
    acnt = Counter(arr)
    bcnt = Counter(brr)
    spare = 0
    for el in acnt.items():
        if el[0] in bcnt:
            get = bcnt[el[0]]
            res += min(el[1], get)
        else:
            spare += el[1]
    if spare:
        res += 1
    else:
        res -= 1
    return res
if __name__ == "__main__":
    n = int(input().strip())
    A = list(map(int, input().strip().split(' ')))
    B = list(map(int, input().strip().split(' ')))
    result = beautifulPairs(A, B)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def hasPathSum(self, root, sum):
        if not root:
            return False
        sum -= root.val
        if sum == 0 and not root.left and not root.right:   
            return True
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Codec:
    def serialize(self, root):
        serial = []
        def preorder(node):
            if not node:
                return
            serial.append(str(node.val))
            for child in node.children:
                preorder(child)
            serial.append("
        preorder(root)
        return " ".join(serial)
    def deserialize(self, data):
        if not data:
            return None
        tokens = deque(data.split())
        root = Node(int(tokens.popleft()), [])
        def helper(node):
            if not tokens:
                return
            while tokens[0] != "
                value = tokens.popleft()
                child = Node(int(value), [])
                node.children.append(child)
                helper(child)
            tokens.popleft()        
        helper(root)
        return root
EOF
class Solution(object):
    def arrangeCoins(self, n):
        return int((math.sqrt(8*n+1)-1) / 2)  
class Solution2(object):
    def arrangeCoins(self, n):
        def check(mid, n):
            return mid*(mid+1) <= 2*n
        left, right = 1, n
        while left <= right:
            mid = left + (right-left)//2
            if not check(mid, n):
                right = mid-1
            else:
                left = mid+1
        return right
EOF
def solve(arr):
    res = 'NO'
    right = sum(arr)
    left = 0
    for el in arr:
        right -= el
        if right == left:
            res = 'YES'
            break
        left += el
    return res
T = int(input().strip())
for a0 in range(T):
    n = int(input().strip())
    a = list(map(int, input().strip().split(' ')))
    result = solve(a)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def basicCalculatorIV(self, expression, evalvars, evalints):
        class CounterMul(Counter):
            def __add__(self, other):
                self.update(other)
                return self
            def __sub__(self, other):
                self.subtract(other)
                return self
            def __mul__(self, other):
                product = CounterMul()
                for x in self:
                    for y in other:
                        xy = tuple(sorted(x + y))
                        product[xy] += self[x] * other[y]
                return product
        vals = dict(zip(evalvars, evalints))        
        def make_counter(token):
            token = str(vals.get(token, token))     
            if token.isalpha():
                return CounterMul({(token,): 1})    
            return CounterMul({(): int(token)})     
        counter = eval(re.sub('(\w+)', r'make_counter("\1")', expression))
        sorted_terms = sorted(counter, key=lambda x: (-len(x), x))
        result = []
        for term in sorted_terms:
            if counter[term]:                       
                result.append("*".join((str(counter[term]),) + term))
        return result
EOF
class Solution(object):
    def lastSubstring(self, s):
        count = collections.defaultdict(list)
        for i in xrange(len(s)):
            count[s[i]].append(i)
        max_c = max(count.iterkeys())
        starts = {}
        for i in count[max_c]:
            starts[i] = i+1
        while len(starts)-1 > 0:
            lookup = set()
            next_count = collections.defaultdict(list)
            for start, end in starts.iteritems():
                if end == len(s):  
                    lookup.add(start)
                    continue
                next_count[s[end]].append(start)				
                if end in starts:  
                    lookup.add(end)			
            next_starts = {}
            max_c = max(next_count.iterkeys())
            for start in next_count[max_c]:
                if start not in lookup:
                    next_starts[start] = starts[start]+1
            starts = next_starts
        return s[next(starts.iterkeys()):]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def addOneRow(self, root, v, d):
        if not root:
            return None
        if d == 1:              
            root, root.left = TreeNode(v), root
        elif d == 2:            
            old_left, old_right = root.left, root.right
            root.left, root.right = TreeNode(v), TreeNode(v)
            root.left.left, root.right.right = old_left, old_right
        else:                   
            self.addOneRow(root.left, v, d - 1)
            self.addOneRow(root.right, v, d - 1)
        return root
EOF
if __name__ == '__main__':
    N = int(input())
    outlist = []
    for _ in range(N):
        args = input().strip().split(' ')
        cmd = args[0]
        if cmd == 'insert':
            outlist.insert(int(args[1]), int(args[2]))
        elif cmd == 'remove':
            outlist.remove(int(args[1]))
        elif cmd == 'append':
            outlist.append(int(args[1]))
        elif cmd == 'print':
            print(outlist)
        elif cmd == 'sort':
            outlist.sort()
        elif cmd == 'pop':
            outlist.pop()
        elif cmd == 'reverse':
            outlist.reverse()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def nextGreaterElement(self, n):
        num = [c for c in str(n)]  
        i = len(num) - 1  
        while i > 0 and num[i - 1] >= num[i]:
            i -= 1
        if i == 0:  
            return -1
        j = i  
        while j + 1 < len(num) and num[j + 1] > num[i - 1]:
            j += 1
        num[j], num[i - 1] = num[i - 1], num[j]  
        result = int("".join(num[:i] + sorted(num[i:])))  
        return -1 if result >= 2 ** 31 else result
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def findMergeNode(head1, head2):
    slow = head1
    fast = head2
    while slow != fast:
        slow = slow.next if slow.next else head2
        fast = fast.next if fast.next else head1
    return slow.data
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        index = int(input())
        llist1_count = int(input())
        llist1 = SinglyLinkedList()
        for _ in range(llist1_count):
            llist1_item = int(input())
            llist1.insert_node(llist1_item)
        llist2_count = int(input())
        llist2 = SinglyLinkedList()
        for _ in range(llist2_count):
            llist2_item = int(input())
            llist2.insert_node(llist2_item)
        ptr1 = llist1.head;
        ptr2 = llist2.head;
        for i in range(llist1_count):
            if i < index:
                ptr1 = ptr1.next
        for i in range(llist2_count):
            if i != llist2_count-1:
                ptr2 = ptr2.next
        ptr2.next = ptr1
        result = findMergeNode(llist1.head, llist2.head)
        fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lengthOfLongestSubstringTwoDistinct(self, s):
        start, max_substring = 0, 0
        last_seen = {}                  
        for i, c in enumerate(s):
            if c in last_seen or len(last_seen) < 2:    
                max_substring = max(max_substring, i - start + 1)
            else:           
                for seen in last_seen:
                    if seen != s[i-1]:              
                       start = last_seen[seen] + 1  
                       del last_seen[seen]
                       break
            last_seen[c] = i
        return max_substring
EOF
def staircase(n):
    for i in range(n):
        for j in range(1, n+1):
            if i + j >= n:
                print("
            else:
                print(" ", end='')
        print()
if __name__ == "__main__":
    n = int(input().strip())
    staircase(n)
EOF
class Solution(object):
    def intToRoman(self, num):
        numeral_map = {1: "I", 4: "IV", 5: "V", 9: "IX", \
                       10: "X", 40: "XL", 50: "L", 90: "XC", \
                       100: "C", 400: "CD", 500: "D", 900: "CM", \
                       1000: "M"}
        keyset, result = sorted(numeral_map.keys()), []
        while num > 0:
            for key in reversed(keyset):
                while num / key > 0:
                    num -= key
                    result += numeral_map[key]
        return "".join(result)
EOF
def repeatedString(s, n):
    return s.count('a') * (n//len(s)) + s[:n%len(s)].count('a')
if __name__ == "__main__":
    s = input().strip()
    n = int(input().strip())
    result = repeatedString(s, n)
    print(result)
EOF
class Solution(object):
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
        for i in xrange(len(strs[0])):
            for string in strs[1:]:
                if i >= len(string) or string[i] != strs[0][i]:
                    return strs[0][:i]
        return strs[0]
class Solution2(object):
    def longestCommonPrefix(self, strs):
        prefix = ""
        for chars in zip(*strs):
            if all(c == chars[0] for c in chars):
                prefix += chars[0]
            else:
                return prefix
        return prefix
EOF
def jump(c):
    res = 0
    ind = 0
    while ind != len(c)-1:
        if ind != len(c)-2 and c[ind+2] == 0:
            ind += 2
        else:
            ind += 1
        res += 1
    return res
if __name__ == "__main__":
    n = int(input().strip())
    c = list(map(int, input().strip().split(' ')))
    result = jump(c)
    print(result)
EOF
def max_permutation(n, k):
    out = []
    switch = k
    if k == 0:
        return [x for x in range(1, n+1)]
    if n % (2*k) != 0:
        return [-1]
    for pos in range(1, n + 1):
        out.append(pos + switch)
        if pos % k == 0:
            switch *= -1
    return out
t = int(input().strip())
for a0 in range(t):
    n, k = input().strip().split(' ')
    n, k = [int(n),int(k)]
    print(" ".join(list(map(str, max_permutation(n, k)))))
EOF
def bigSorting(arr):
    return sorted(arr, key=int)
if __name__ == "__main__":
    n = int(input().strip())
    arr = []
    arr_i = 0
    for arr_i in range(n):
        arr_t = input().strip()
        arr.append(arr_t)
    result = bigSorting(arr)
    print ("\n".join(map(str, result)))
EOF
class Solution(object):
    def sumOfDigits(self, A):
        total = sum([int(c) for c in str(min(A))])
        return 1 if total % 2 == 0 else 0
EOF
input()
shoes = list(map(int,input().split()))
shoesCollection = collections.Counter(shoes)
sale = 0
for i in range(int(input())):
    s,n = map(int,input().split())
    if shoesCollection[s] > 0:
        sale += n
        shoesCollection[s] -= 1
print(sale)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sortArrayByParity(self, A):
        return [num for num in A if num % 2 == 0] + [num for num in A if num % 2 == 1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findCheapestPrice(self, n, flights, src, dst, K):
        flts = defaultdict(list)                    
        for start, end, cost in flights:
            flts[start].append((end, cost))
        queue = [(0, -1, src)]                      
        visited = set()                             
        while queue:
            cost, stops, location = heapq.heappop(queue)
            visited.add(location)
            if location == dst:
                return cost
            if stops == K:                          
                continue
            for end, next_cost in flts[location]:
                if end not in visited:              
                    heapq.heappush(queue, (cost + next_cost, stops + 1, end))
        return -1                                   
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeLinkNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None
class Solution(object):
    def connect(self, root):
        level = [root]
        while level and level[0]:   
            next_level = []
            prev = None
            for node in level:      
                if prev:
                    prev.next = node
                prev = node
                next_level.append(node.left)    
                next_level.append(node.right)
            level = next_level
EOF
for _ in range(int(input())):
    s = input()
    print(s[0::2],s[1::2])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isCousins(self, root, x, y):
        val_to_node = {root.val: root}  
        node_to_parent = {root: None}
        while True:
            x_node = val_to_node.get(x, None)
            y_node = val_to_node.get(y, None)
            if x_node is not None and y_node is not None:
                return node_to_parent[x_node] != node_to_parent[y_node]
            if x_node is not None or y_node is not None:
                return False
            new_val_to_node = {}
            for node in val_to_node.values():
                if node.left:
                    node_to_parent[node.left] = node
                    new_val_to_node[node.left.val] = node.left
                if node.right:
                    node_to_parent[node.right] = node
                    new_val_to_node[node.right.val] = node.right
            val_to_node = new_val_to_node
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def subsets(self, nums):
        nb_subsets = 2**len(nums)
        all_subsets = []
        for subset_nb in range(nb_subsets):
            subset = []
            for num in nums:
                if subset_nb % 2 == 1:
                    subset.append(num)
                subset_nb //= 2
            all_subsets.append(subset)
        return all_subsets
EOF
class OrderedCounter(Counter, OrderedDict):
    pass
d = OrderedCounter(input() for _ in range(int(input())))
print(len(d))
print(*d.values())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def numTrees(self, n):
        memo = [-1] * (n+1)     
        return self.helper(n, memo)
    def helper(self, n, memo):
        if n <= 1:
            return 1    
        if memo[n] != -1:
            return memo[n]
        count = 0
        for i in range(1, n+1):     
            count += self.helper(i-1, memo) * self.helper(n-i, memo)
        memo[n] = count
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def divide(self, dividend, divisor):
        if divisor == 0:
            return None
        diff_sign = (divisor < 0) ^ (dividend < 0)
        dividend = abs(dividend)
        divisor = abs(divisor)
        result = 0
        max_divisor = divisor
        shift_count = 1
        while dividend >= (max_divisor << 1):   
            max_divisor <<= 1
            shift_count <<= 1
        while shift_count >= 1:
            if dividend >= max_divisor:         
                dividend -= max_divisor
                result += shift_count
            shift_count >>= 1
            max_divisor >>= 1
        if diff_sign:
            result = -result
        return max(min(result, 2**31-1), -2**31)        
EOF
Regex_Pattern = r'hackerrank'	
Test_String = input()
match = re.findall(Regex_Pattern, Test_String)
print("Number of matches :", len(match))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def makesquare(self, nums):
        def dfs(index):
            if index == len(nums):              
                return True
            for side in range(4):
                if sides[side] + nums[index] > target or sides[side] in sides[side + 1:]:   
                    continue                    
                sides[side] += nums[index]      
                if dfs(index + 1):
                    return True
                sides[side] -= nums[index]      
            return False                        
        perimeter = sum(nums)
        target, remainder = divmod(perimeter, 4)
        if not perimeter or remainder:          
            return False
        nums.sort(reverse = True)               
        if nums[0] > target:                    
            return False
        sides = [0] * 4                         
        i = 0
        while i < 4 and nums[i] == target:      
            sides[i] = target
            i += 1
        return dfs(i)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def accountsMerge(self, accounts):
        email_to_account = defaultdict(list)        
        for i, account in enumerate(accounts):
            for email in account[1:]:
                email_to_account[email].append(i)
        result = []
        visited = [False for _ in range(len(accounts))]
        def dfs(i):
            emails = set()
            if visited[i]:
                return emails
            visited[i] = True
            for email in accounts[i][1:]:
                emails.add(email)
                for account in email_to_account[email]:
                    emails |= dfs(account)          
            return emails
        for i, account in enumerate(accounts):
            emails = dfs(i)
            if emails:
                result.append([account[0]] + sorted(list(emails)))
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        return root
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def uniquePathsIII(self, grid):
        rows, cols = len(grid), len(grid[0])
        unvisited = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    start = (r, c)
                elif grid[r][c] == 2:
                    end = (r, c)
                    unvisited.add((r, c))
                elif grid[r][c] == 0:
                    unvisited.add((r, c))
        def make_paths(r, c):
            if not unvisited and (r, c) == end:     
                return 1
            if not unvisited or (r, c) == end:      
                return 0
            paths = 0
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nbor_r, nbor_c = r + dr, c + dc
                if (nbor_r, nbor_c) in unvisited:
                    unvisited.remove((nbor_r, nbor_c))
                    paths += make_paths(nbor_r, nbor_c)
                    unvisited.add((nbor_r, nbor_c)) 
            return paths
        return make_paths(*start)
EOF
class Person:
    def __init__(self, initialAge):
        if initialAge < 0:
            self.age = 0
            print('Age is not valid, setting age to 0.')
        else:
            self.age = initialAge
    def yearPasses(self):
        self.age += 1
    def amIOld(self):
        if self.age < 13:
            print('You are young.')
        elif self.age >= 13 and self.age < 18:
            print('You are a teenager.')
        else:
            print('You are old.')
t = int(input())
for i in range(0, t):
    age = int(input())
    p = Person(age)
    p.amIOld()
    for j in range(0, 3):
        p.yearPasses()
    p.amIOld()
    print("")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, w):
        self.cumulative = []
        total = 0
        for weight in w:
            total += weight
            self.cumulative.append(total)
    def pickIndex(self):
        x = random.randint(1, self.cumulative[-1])
        return bisect.bisect_left(self.cumulative, x)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def serialize(self, root):
    serial_list = []
    def serial(node):
        if not node:
            return
        serial_list.append(str(node.val))
        serial(node.left)
        serial(node.right)
    serial(root)
    return " ".join(serial_list)
def deserialize(self, data):
    preorder = deque(int(val) for val in data.split())  
    def deserial(low, high):
        if preorder and low < preorder[0] < high:
            val = preorder.popleft()
            node = TreeNode(val)
            node.left = deserial(low, val)
            node.right = deserial(val, high)
            return node
    return deserial(float("-inf"), float("inf"))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProfit(self, prices):
        buy1, buy2 = float('-inf'), float('-inf')
        sell1, sell2 = 0, 0
        for price in prices:
            buy1 = max(buy1, -price)
            sell1 = max(sell1, price + buy1)
            buy2 = max(buy2, sell1 - price)
            sell2 = max(sell2, price + buy2)
        return sell2
EOF
if __name__ == '__main__':
    x,y,z, n = [int(input()) for i in range(4)]
    print([[i,j,k] for i in range(0, x + 1) for j in range(0, y + 1) for k in range(0, z+ 1) if(i + j + k) != n])
EOF
class Person:
	def __init__(self, firstName, lastName, idNumber):
		self.firstName = firstName
		self.lastName = lastName
		self.idNumber = idNumber
	def printPerson(self):
		print("Name:", self.lastName + ",", self.firstName)
		print("ID:", self.idNumber)
class Student(Person):
    def __init__(self, firstName, lastName, idNumber,scores):
        self.firstName = firstName
        self.lastName = lastName
        self.idNumber = idNumber
        self.scores = scores
    def calculate(self):
        r = sum(self.scores)/len(self.scores)
        if r >= 90:
            return 'O'
        elif r >=80 and r < 90:
            return 'E'
        elif r >= 70 and r < 80:
            return 'A'
        elif r >= 55 and r < 70:
            return 'P'
        elif r >= 40 and r < 55:
            return 'D'
        else:
            return 'T'
line = input().split()
firstName = line[0]
lastName = line[1]
idNum = line[2]
numScores = int(input()) 
scores = list( map(int, input().split()) )
s = Student(firstName, lastName, idNum, scores)
s.printPerson()
print("Grade:", s.calculate())
EOF
for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a // b)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)
EOF
len_m, m = input(), set(map(int, input().split()))
len_n, n = input(), set(map(int, input().split()))
print(*sorted((m.symmetric_difference(n))), sep='\n')
EOF
class AdvancedArithmetic(object):
    def divisorSum(n):
        raise NotImplementedError
class Calculator(AdvancedArithmetic):
    def divisorSum(self, n):
        res = 0
        for i in range(1,n+1):
            if n % i == 0:
                res += i
        return res
n = int(input())
my_calculator = Calculator()
s = my_calculator.divisorSum(n)
print("I implemented: " + type(my_calculator).__bases__[0].__name__)
print(s)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def hammingDistance(self, x, y):
        hamming = 0
        while x or y:
            hamming += (x & 1) != (y & 1)
            x >>= 1
            y >>= 1
        return hamming
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyCalendarTwo(object):
    def __init__(self):
        self.doubles = []       
        self.intervals = []     
    def book(self, start, end):
        for i, j in self.doubles:   
            if start < j and end > i:
                return False
        for i, j in self.intervals: 
            if start < j and end > i:
                self.doubles.append((max(start, i), min(end, j)))
        self.intervals.append((start, end))     
        return True
EOF
if __name__ == "__main__":
    t = int(input().strip())
    for _ in range(t):
        a, b = input().strip().split(' ')
        try:
            print(int(a)//int(b))
        except ZeroDivisionError as e:
            print("Error Code: {}".format(e))
        except ValueError as e:
            print("Error Code: {}".format(e))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canReorderDoubled(self, A):
        counts = Counter(A)
        for num in sorted(counts, key=abs):     
            if counts[num] > counts[num * 2]:
                return False
            counts[num * 2] -= counts[num]
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numberToWords(self, num):
        int_to_word = {1 : 'One', 2 : 'Two', 3 : 'Three', 4 : 'Four', 5 : 'Five',
                    6 : 'Six', 7 : 'Seven', 8 : 'Eight', 9 : 'Nine', 10 : 'Ten',
                    11 : 'Eleven', 12 : 'Twelve', 13 : 'Thirteen', 14 : 'Fourteen',
                    15 : 'Fifteen', 16 : 'Sixteen', 17 : 'Seventeen', 18 : 'Eighteen',
                    19: 'Nineteen', 20 : 'Twenty', 30 : 'Thirty', 40 : 'Forty',
                    50 : 'Fifty', 60 : 'Sixty', 70 : 'Seventy', 80 : 'Eighty', 90 : 'Ninety'}
        digits_to_word = {3 : 'Thousand', 6 : 'Million', 9 : 'Billion', 12 : 'Trillion',
                    15 : 'Quadrillion', 18 : 'Quintillion', 21 : 'Sextillion', 24 : 'Septillion',
                    27 : 'Octillion', 30 : 'Nonillion'}
        english = deque()
        digits = 0
        if num == 0:
            return "Zero"
        while num:
            num, section = divmod(num, 1000)        
            hundreds, tens = divmod(section, 100)
            if section and digits > 0:
                english.appendleft(digits_to_word[digits])
            digits += 3
            if tens >= 20:
                if tens%10:
                    english.appendleft(int_to_word[tens%10])
                english.appendleft(int_to_word[10*(tens//10)])
            elif tens:
                english.appendleft(int_to_word[tens])
            if hundreds:
                english.appendleft("Hundred")
                english.appendleft(int_to_word[hundreds])
        return " ".join(english)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def intersectionSizeTwo(self, intervals):
        intervals.sort(key=lambda x: x[1])
        intersection = []
        for start, end in intervals:
            if not intersection or start > intersection[-1]:
                intersection.append(end - 1)
                intersection.append(end)
            elif start > intersection[-2]:
                intersection.append(end)
        return len(intersection)
EOF
def compareTriplets(a, b):
    alice,bob =0,0
    for i in range(3):
        if a[i] > b[i]:
            alice += 1
        elif a[i] < b[i]:
            bob += 1
    return [alice,bob] 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    a = list(map(int, input().rstrip().split()))
    b = list(map(int, input().rstrip().split()))
    result = compareTriplets(a, b)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
EOF
def countingSort(arr):
    cnt = [0] * (max(arr) + 1)
    output = [0] * (len(arr))
    for el in arr:
        cnt[el] += 1
    total = 0
    for ind in range(len(cnt)):
        old = cnt[ind]
        cnt[ind] = total
        total += old
    for el in arr:
        output[cnt[el]] = el
        cnt[el] += 1
    return output
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = countingSort(arr)
    print (" ".join(map(str, result)))
EOF
def hackerrankInString(s):
    hckr = list('hackerrank')
    index = 0
    res = ''
    for let in s:
        if index == len(hckr):
            break
        if let == hckr[index]:
            index += 1
    if index == len(hckr):
        res = 'YES'
    else:
        res = 'NO'
    return res
if __name__ == "__main__":
    q = int(input().strip())
    for a0 in range(q):
        s = input().strip()
        result = hackerrankInString(s)
        print(result)
EOF
def cipher(k, s):
    s_len = len(s)
    out_len = s_len - k + 1
    result = [0] * out_len
    result[-1] = int(s[-1])
    keep_prev = result[-1]
    for it_ind in range(1, out_len):
        to_update = out_len - 1 - it_ind
        result[to_update] = int(s[-(it_ind+1)])
        if it_ind >= k:
            keep_prev ^= result[to_update + k]
        result[to_update] ^= keep_prev
        keep_prev ^= result[to_update]
    return "".join(map(str, result))
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    s = input().strip()
    result = cipher(k, s)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def solveSudoku(self, board):
        self.size = 9
        self.board = board
        self.new_digits = []        
        for r in range(self.size):
            self.board[r] = [digit for digit in self.board[r]]          
            for c in range(self.size):
                if self.board[r][c] == '.':
                    self.board[r][c] = {str(i) for i in range(1, 10)}   
                else:
                    self.new_digits.append((r, c))
        while self.new_digits:
            for r, c in self.new_digits:
                self.eliminate(r, c)        
                self.new_digits = []
                self.find_new()             
        self.solve_recursive()
    def eliminate(self, row, col):
        for i in range(self.size):
            if isinstance(self.board[i][col], set):
                self.board[i][col].discard(self.board[row][col])   
            if isinstance(self.board[row][i], set):
                self.board[row][i].discard(self.board[row][col])
        for box_row in range(3*(row//3), 3 + 3*(row//3)):
            for box_col in range(3*(col//3), 3 + 3*(col//3)):
                if isinstance(self.board[box_row][box_col], set):
                    self.board[box_row][box_col].discard(self.board[row][col])
    def find_new(self):
        for row in range(self.size):
            for col in range(self.size):
                if isinstance(self.board[row][col], set) and len(self.board[row][col]) == 1:
                    self.board[row][col] = self.board[row][col].pop()
                    self.new_digits.append((row,col))
    def solve_recursive(self):
        for r in range(self.size):
            for c in range(self.size):
                if len(self.board[r][c]) == 1:
                    continue
                for digit in self.board[r][c]:          
                    if self.is_valid(r, c, digit):      
                        save_set = self.board[r][c]
                        self.board[r][c] = digit
                        if self.solve_recursive():
                            return True
                        self.board[r][c] = save_set     
                return False
        return True
    def is_valid(self, row, col, digit):    
        for i in range(self.size):          
            if self.board[row][i] == digit or self.board[i][col] == digit:
                return False
        n = self.size // 3
        for r in range(n*(row//n), n + n*(row//n)):     
            for c in range(n*(col//n), n + n*(col//n)):
                if self.board[r][c] == digit:
                    return False
        return True
EOF
class Solution(object):
    def kWeakestRows(self, mat, k):
        result, lookup = [], set()
        for j in xrange(len(mat[0])):
            for i in xrange(len(mat)):
                if mat[i][j] or i in lookup:
                    continue
                lookup.add(i)
                result.append(i)
                if len(result) == k:
                    return result
        for i in xrange(len(mat)):
            if i in lookup:
                continue
            lookup.add(i)
            result.append(i)
            if len(result) == k:
                break
        return result
class Solution2(object):
    def kWeakestRows(self, mat, k):
        lookup = collections.OrderedDict()
        for j in xrange(len(mat[0])):
            for i in xrange(len(mat)):
                if mat[i][j] or i in lookup:
                    continue
                lookup[i] = True
                if len(lookup) == k:
                    return lookup.keys()
        for i in xrange(len(mat)):
            if i in lookup:
                continue
            lookup[i] = True
            if len(lookup) == k:
                break
        return lookup.keys()
class Solution3(object):
    def kWeakestRows(self, mat, k):
        def nth_element(nums, n, compare=lambda a, b: a < b):
            def partition_around_pivot(left, right, pivot_idx, nums, compare):
                new_pivot_idx = left
                nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
                for i in xrange(left, right):
                    if compare(nums[i], nums[right]):
                        nums[i], nums[new_pivot_idx] = nums[new_pivot_idx], nums[i]
                        new_pivot_idx += 1
                nums[right], nums[new_pivot_idx] = nums[new_pivot_idx], nums[right]
                return new_pivot_idx
            left, right = 0, len(nums) - 1
            while left <= right:
                pivot_idx = random.randint(left, right)
                new_pivot_idx = partition_around_pivot(left, right, pivot_idx, nums, compare)
                if new_pivot_idx == n:
                    return
                elif new_pivot_idx > n:
                    right = new_pivot_idx - 1
                else:  
                    left = new_pivot_idx + 1
        nums = [(sum(mat[i]), i) for i in xrange(len(mat))]
        nth_element(nums, k)
        return map(lambda x: x[1], sorted(nums[:k]))
EOF
def FindMergeNode(headA, headB):
    values_a = []
    values_b = []
    node_a = headA
    node_b = headB
    while node_a != None:
        values_a.append(node_a.data)
        node_a = node_a.next
    while node_b != None:
        values_b.append(node_b.data)
        node_b = node_b.next
    res = values_a[-1]
    ind = 1
    while values_a[-ind] == values_b[-ind]:
        ind += 1
    res = values_a[-(ind-1)]
    return res
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def selfDividingNumbers(self, left, right):
        def is_self_dividing(num):
            copy = num              
            while copy > 0:
                copy, digit = divmod(copy, 10)
                if digit == 0 or num % digit != 0:
                    return False
            return True
        return [num for num in range(left, right + 1) if is_self_dividing(num)]
EOF
class Solution(object):
    def largestNumber(self, cost, target):
        dp = [0]
        for t in xrange(1, target+1):
            dp.append(-1)
            for i, c in enumerate(cost):
                if t-c < 0 or dp[t-c] < 0:
                    continue
                dp[t] = max(dp[t], dp[t-c]+1)
        if dp[target] < 0:
            return "0"
        result = []
        for i in reversed(xrange(9)):
            while target >= cost[i] and dp[target] == dp[target-cost[i]]+1:
                target -= cost[i]
                result.append(i+1)
        return "".join(map(str, result))
class Solution2(object):
    def largestNumber(self, cost, target):
        def key(bag):
            return sum(bag), bag
        dp = [[0]*9]
        for t in xrange(1, target+1):
            dp.append([])
            for d, c in enumerate(cost):
                if t < c or not dp[t-c]:
                    continue
                curr = dp[t-c][:]
                curr[~d] += 1
                if key(curr) > key(dp[t]):
                    dp[-1] = curr        
        if not dp[-1]:
            return "0"
        return "".join(str(9-i)*c for i, c in enumerate(dp[-1]))
class Solution3(object):
    def largestNumber(self, cost, target):
        dp = [0]
        for t in xrange(1, target+1):
            dp.append(-1)
            for i, c in enumerate(cost):
                if t-c < 0:
                    continue
                dp[t] = max(dp[t], dp[t-c]*10 + i+1)
        return str(max(dp[t], 0))
EOF
class Solution(object):
    def maxSubArray(self, nums):
        result, curr = float("-inf"), float("-inf")
        for x in nums:
            curr = max(curr+x, x)
            result = max(result, curr)
        return result
EOF
def countingSort(arr):
    output = [0] * (max(arr) + 1)
    for el in arr:
        output[el] += 1
    return output
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = countingSort(arr)
    print (" ".join(map(str, result)))
EOF
def sockMerchant(n, ar):
    d = {}
    for i in ar:
        d[i] = d.get(i,0) + 1
    res = 0
    for k,v in d.items():
        res += v//2
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = sockMerchant(n, ar)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def licenseKeyFormatting(self, S, K):
        key = S.replace("-", "").upper()
        formatted = []
        i = len(key) - K
        while i >= 0:
            formatted.append(key[i:i + K])
            i -= K
        if i != -K:
            formatted.append(key[:i + K])
        return "-".join(formatted[::-1])
        return "-".join(formatted)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def generateAbbreviations(self, word):
        abbreviations = [[]]
        for c in word:
            new_abbreviations = []
            for abbr in abbreviations:
                if len(abbr) > 0 and isinstance(abbr[-1], int):
                    new_abbreviations.append(abbr[:-1] + [abbr[-1] + 1])
                else:
                    new_abbreviations.append(abbr + [1])
                new_abbreviations.append(abbr + [c])
            abbreviations = new_abbreviations
        return ["".join(map(str, a)) for a in abbreviations]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        def helper(i, j):   
            if i > j:
                return None
            max_num = float("-inf")
            for k in range(i, j + 1):
                if nums[k] > max_num:
                    max_num = nums[k]
                    max_index = k
            root = TreeNode(max_num)
            root.left = helper(i, max_index - 1)
            root.right = helper(max_index + 1, j)
            return root
        return helper(0, len(nums) - 1)
EOF
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        if k <= 1: return 0
        result, start, prod = 0, 0, 1
        for i, num in enumerate(nums):
            prod *= num
            while prod >= k:
                prod /= nums[start]
                start += 1
            result += i-start+1
        return result
EOF
def isValid(s):
    cnt = Counter(s)
    res = 'NO'
    print("cnt = {} len = {}".format(cnt, len(set(cnt.values()))))
    if len(set(cnt.values())) == 1:
        res = 'YES'
    elif len(set(cnt.values())) == 2:
        bigger = max(cnt.values())
        lesser = min(cnt.values())
        bigger_let = [let for let, c in cnt.items() if c == bigger]
        lesser_let = [let for let, c in cnt.items() if c == lesser]
        if len(lesser_let) == 1 and lesser == 1:
            res = 'YES'
        elif len(bigger_let) == 1 or len(lesser_let) == 1:
            if abs(bigger-lesser) == 1:
                res = 'YES'
            else:
                res = 'NO'
        else:
            res = 'NO'
    else:
        res = 'NO'
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = isValid(s)
    fptr.write(result + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, nums):
        self.nums = nums
    def pick(self, target):
        count = 0
        for i, num in enumerate(self.nums):
            if num == target:
                if random.randint(0, count) == 0:
                    result = i
                count += 1
        return result
EOF
def breakingRecords(score):
    arr_max = [score[0]]
    arr_min = [score[0]]
    for el in score[1:]:
        if el < arr_min[-1]:
            arr_min.append(el)
        if el > arr_max[-1]:
            arr_max.append(el)
    return str(len(arr_max)-1), str(len(arr_min)-1)
if __name__ == "__main__":
    n = int(input().strip())
    score = list(map(int, input().strip().split(' ')))
    result = breakingRecords(score)
    print (" ".join(map(str, result)))
EOF
def beautifulBinaryString(b):
    res = 0
    while '010' in b:
        res += 1
        b = b.replace("010","011", 1)
    return res
if __name__ == "__main__":
    n = int(input().strip())
    b = input().strip()
    result = beautifulBinaryString(b)
    print(result)
EOF
class Solution(object):
    def minRefuelStops(self, target, startFuel, stations):
        max_heap = []
        stations.append((target, float("inf")))
        result = prev = 0
        for location, capacity in stations:
            startFuel -= location - prev
            while max_heap and startFuel < 0:
                startFuel += -heapq.heappop(max_heap)
                result += 1
            if startFuel < 0:
                return -1
            heapq.heappush(max_heap, -capacity)
            prev = location
        return result
EOF
class Solution(object):
    def numberWays(self, hats):
        MOD = 10**9 + 7
        HAT_SIZE = 40
        hat_to_people = [[] for _ in xrange(HAT_SIZE)]
        for i in xrange(len(hats)):
            for h in hats[i]:
                hat_to_people[h-1].append(i)
        dp = [0]*(1<<len(hats))
        dp[0] = 1
        for people in hat_to_people:
            for mask in reversed(xrange(len(dp))):
                for p in people:
                    if mask & (1<<p):
                        continue
                    dp[mask | (1<<p)] += dp[mask]
                    dp[mask | (1<<p)] %= MOD
        return dp[-1]
EOF
def largestRectangle(h):
    res = 0
    for i in range(len(h)):
        length = 0
        for j in range(i, -1, -1):
            if h[j] >= h[i]:
                length += 1
            else:
                break
        for j in range(i+1, len(h)):
            if h[j] >= h[i]:
                length += 1
            else:
                break
        res = max(res, length*h[i])
    return res
if __name__ == "__main__":
    n = int(input().strip())
    h = list(map(int, input().strip().split(' ')))
    result = largestRectangle(h)
    print(result)
EOF
class Solution(object):
    def countSmaller(self, nums):
        def countAndMergeSort(num_idxs, start, end, counts):
            if end - start <= 0:  
                return 0
            mid = start + (end - start) / 2
            countAndMergeSort(num_idxs, start, mid, counts)
            countAndMergeSort(num_idxs, mid + 1, end, counts)
            r = mid + 1
            tmp = []
            for i in xrange(start, mid + 1):
                while r <= end and num_idxs[r][0] < num_idxs[i][0]:
                    tmp.append(num_idxs[r])
                    r += 1
                tmp.append(num_idxs[i])
                counts[num_idxs[i][1]] += r - (mid + 1)
            num_idxs[start:start+len(tmp)] = tmp
        num_idxs = []
        counts = [0] * len(nums)
        for i, num in enumerate(nums):
            num_idxs.append((num, i))
        countAndMergeSort(num_idxs, 0, len(num_idxs) - 1, counts)
        return counts
class Solution2(object):
    def countSmaller(self, nums):
        def binarySearch(A, target, compare):
            start, end = 0, len(A) - 1
            while start <= end:
                mid = start + (end - start) / 2
                if compare(target, A[mid]):
                    end = mid - 1
                else:
                    start = mid + 1
            return start
        class BIT(object):
            def __init__(self, n):
                self.__bit = [0] * n
            def add(self, i, val):
                while i < len(self.__bit):
                    self.__bit[i] += val
                    i += (i & -i)
            def query(self, i):
                ret = 0
                while i > 0:
                    ret += self.__bit[i]
                    i -= (i & -i)
                return ret
        sorted_nums, places = sorted(nums), [0] * len(nums)
        for i, num in enumerate(nums):
            places[i] = binarySearch(sorted_nums, num, lambda x, y: x <= y)
        ans, bit = [0] * len(nums), BIT(len(nums) + 1)
        for i in reversed(xrange(len(nums))):
            ans[i] = bit.query(places[i])
            bit.add(places[i] + 1, 1)
        return ans
class Solution3(object):
    def countSmaller(self, nums):
        res = [0] * len(nums)
        bst = self.BST()
        for i in reversed(xrange(len(nums))):
            bst.insertNode(nums[i])
            res[i] = bst.query(nums[i])
        return res
    class BST(object):
        class BSTreeNode(object):
            def __init__(self, val):
                self.val = val
                self.count = 0
                self.left = self.right = None
        def __init__(self):
            self.root = None
        def insertNode(self, val):
            node = self.BSTreeNode(val)
            if not self.root:
                self.root = node
                return
            curr = self.root
            while curr:
                if node.val < curr.val:
                    curr.count += 1  
                    if curr.left:
                        curr = curr.left
                    else:
                        curr.left = node
                        break
                else:  
                    if curr.right:
                        curr = curr.right
                    else:
                        curr.right = node
                        break
        def query(self, val):
            count = 0
            curr = self.root
            while curr:
                if val < curr.val:
                    curr = curr.left
                elif val > curr.val:
                    count += 1 + curr.count  
                    curr = curr.right
                else:  
                    return count + curr.count
            return 0
EOF
def lcs(X , Y): 
    m = len(X) 
    n = len(Y) 
    L = [[None]*(n+1) for i in range(m+1)] 
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
    return L[m][n]
def commonChild(s1, s2):
    common_letters = set(s1) & set(s2)
    print("intersect: {}".format(common_letters))
    if (not bool(common_letters)):
        return 0
    s1_filt = "".join([x for x in s1 if x in common_letters])
    s2_filt = "".join([x for x in s2 if x in common_letters])
    print("s1_filt: {}".format(s1_filt))
    print("s2_filt: {}".format(s2_filt))
    return lcs(s1, s2)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s1 = input()
    s2 = input()
    result = commonChild(s1, s2)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
def find_plus(grid, x, y, mx):
    n = len(grid)
    m = len(grid[0])
    deep = 0
    no_more = 0
    i = 1
    if grid[x][y] != 1:
        return 0, no_more
    while (x+i < n and x-i >= 0 and y+i < m and y-i >= 0 and
          grid[x+i][y] == 1 and grid[x-i][y] == 1 and
          grid[x][y+i] == 1 and grid[x][y-i] == 1 and
          deep < mx):
        grid[x+i][y] = 2
        grid[x-i][y] = 2
        grid[x][y+i] = 2
        grid[x][y-i] = 2
        i += 1
        deep += 1
    if i > 1:
        grid[x][y] = 2
    if mx >= i:
        no_more = 1
    return 1 + 4*(i-1), no_more
def find_pair(grid, x, y):
    res = 1
    best_x = 0
    best_y = 0
    for ind in range(x, n):
        for jnd in range(m):
            if ind == x and jnd <= y:
                continue
            matrix = copy.deepcopy(grid)
            temp_res, ign = find_plus(matrix, ind, jnd, max(n, m)+1)
            if temp_res > res:
                res = max(res, temp_res)
                best_x = ind
                best_y = jnd
    return res, best_x, best_y
def twoPluses(grid):
    res = 0
    n = len(grid)
    m = len(grid[0])
    b_x_1 = 0
    b_y_1 = 0
    b_x_2 = 0
    b_y_2 = 0
    b_1 = 0
    b_2 = 0
    for row in grid:
        for el in row:
            if el == 1:
                res = 1
                break
    for ind in range(n):
        for jnd in range(m):
            for mx in range(max(n, m)+1):
                matrix = copy.deepcopy(grid)
                first, skip = find_plus(matrix, ind, jnd, mx)
                if first == 0 or skip == 1:
                    continue
                second, t_x, t_y = find_pair(matrix, ind, jnd)
                if first * second > res:
                    res = max(res, first*second)
                    b_x_1 = ind
                    b_y_1 = jnd
                    b_1 = first
                    b_x_2 = t_x
                    b_y_2 = t_y
                    b_2 = second
    return res
def print_grid(grid):
    for row in grid:
        for el in row:
            print(el, end='')
        print()
if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    grid = []
    grid_i = 0
    for grid_i in range(n):
        grid_t = str(input().strip())
        grid.append(grid_t)
    grid_norm = [[0]*m for _ in range(n)]
    for ind in range(n):
        for jnd in range(m):
            if grid[ind][jnd] == 'G':
                grid_norm[ind][jnd] = 1
    result = twoPluses(grid_norm)
    print(result)
EOF
def jumpingOnClouds(c, k):
    cur = k % n
    energy = 100 - 1 - c[cur]*2
    while cur != 0:
        cur = (cur + k) % n
        energy -= 1 + c[cur]*2
    return energy
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    c = list(map(int, input().strip().split(' ')))
    result = jumpingOnClouds(c, k)
    print(result)
EOF
def check(array):
    return len(list(filter(lambda x: x%2 == 0, array))) == len(array)
def fairRations(B):
    res = 0
    for ind in range(len(B)-1):
        if B[ind]%2 == 1:
            B[ind] += 1
            B[ind+1] += 1
            res += 2
    return res if check(B) else 'NO'
if __name__ == "__main__":
    N = int(input().strip())
    B = list(map(int, input().strip().split(' ')))
    result = fairRations(B)
    print(result)
EOF
class Solution(object):
    def solveNQueens(self, n):
        def dfs(curr, cols, main_diag, anti_diag, result):
            row, n = len(curr), len(cols)
            if row == n:
                result.append(map(lambda x: '.'*x + "Q" + '.'*(n-x-1), curr))
                return
            for i in xrange(n):
                if cols[i] or main_diag[row+i] or anti_diag[row-i+n]:
                    continue
                cols[i] = main_diag[row+i] = anti_diag[row-i+n] = True
                curr.append(i)
                dfs(curr, cols, main_diag, anti_diag, result)
                curr.pop()
                cols[i] = main_diag[row+i] = anti_diag[row-i+n] = False
        result = []
        cols, main_diag, anti_diag = [False]*n, [False]*(2*n), [False]*(2*n)
        dfs([], cols, main_diag, anti_diag, result)
        return result
class Solution2(object):
    def solveNQueens(self, n):
        def dfs(col_per_row, xy_diff, xy_sum):
            cur_row = len(col_per_row)
            if cur_row == n:
                ress.append(col_per_row)
            for col in range(n):
                if col not in col_per_row and cur_row-col not in xy_diff and cur_row+col not in xy_sum:
                    dfs(col_per_row+[col], xy_diff+[cur_row-col], xy_sum+[cur_row+col])
        ress = []
        dfs([], [], [])
        return [['.'*i + 'Q' + '.'*(n-i-1) for i in res] for res in ress]
EOF
int main() {
    int T;
    unsigned int number;
    unsigned int mask = -1;
    scanf("%d", &T);
    for (int i = 0; i < T; i++) {
        scanf("%u", &number);
        printf("%u\n", number^mask);
    }
    return 0;
}
EOF
class Solution(object):
    def findMinFibonacciNumbers(self, k):
        result, a, b = 0, 1, 1
        while b <= k:
            b, a = a+b, b
        while k:
            if a <= k:
                k -= a
                result += 1
            a, b = b-a, a
        return result
EOF
class Leaderboard(object):
    def __init__(self):
        self.__lookup = collections.Counter()
    def addScore(self, playerId, score):
        self.__lookup[playerId] += score
    def top(self, K):
        def kthElement(nums, k, compare):
            def PartitionAroundPivot(left, right, pivot_idx, nums, compare):
                new_pivot_idx = left
                nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
                for i in xrange(left, right):
                    if compare(nums[i], nums[right]):
                        nums[i], nums[new_pivot_idx] = nums[new_pivot_idx], nums[i]
                        new_pivot_idx += 1
                nums[right], nums[new_pivot_idx] = nums[new_pivot_idx], nums[right]
                return new_pivot_idx
            left, right = 0, len(nums) - 1
            while left <= right:
                pivot_idx = random.randint(left, right)
                new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums, compare)
                if new_pivot_idx == k:
                    return
                elif new_pivot_idx > k:
                    right = new_pivot_idx - 1
                else:  
                    left = new_pivot_idx + 1
        scores = self.__lookup.values()
        kthElement(scores, K, lambda a, b: a > b)
        return sum(scores[:K])
    def reset(self, playerId):
        self.__lookup[playerId] = 0
EOF
def equalize(arr):
    res = 0
    for el in arr:
        while el != 0:
            if el == 1 or el == 2 or el == 5:
                el -= el
                res += 1
            elif el == 3 or el == 4:
                el -= el
                res += 2
            else:
                res += el//5
                el = int(el%5)
    return res
def equal(arr):
    res = 10**8
    arr_sorted = sorted(arr)
    arr_min = min(arr_sorted)
    for ind in range(3):
        arr_new = [ x - arr_min + ind for x in arr_sorted ]
        res = min(res, equalize(arr_new))
    return res
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = int(input().strip())
        arr = list(map(int, input().strip().split(' ')))
        result = equal(arr)
        print(result)
EOF
def biggerIsGreater(w):
    arr = list(w)
    i = len(arr) - 1
    while i > 0 and arr[i - 1] >= arr[i]:
            i -= 1
    if i <= 0:
    	return 'no answer'
    j = len(arr) - 1
    while arr[j] <= arr[i - 1]:
        j -= 1
    arr[i - 1], arr[j] = arr[j], arr[i - 1]
    arr[i : ] = arr[len(arr) - 1 : i - 1 : -1]
    return "".join(arr)
if __name__ == "__main__":
    T = int(input().strip())
    for a0 in range(T):
        w = input().strip()
        result = biggerIsGreater(w)
        print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestOverlap(self, A, B):
        def image_to_bits(image):       
            bits = []
            for row in image:
                num = 0
                for i, bit in enumerate(reversed(row)):
                    if bit == 1:
                        num += (bit << i)
                bits.append(num)
            return bits
        A_bits, B_bits = image_to_bits(A), image_to_bits(B)
        rows, cols = len(A), len(A[0])
        max_overlap = 0
        for slide, static in ((A_bits, B_bits), (B_bits, A_bits)):  
            for row_shift in range(rows):
                for col_shift in range(cols):
                    overlap = 0
                    for slide_row in range(rows - row_shift):       
                        shifted = slide[slide_row] >> col_shift     
                        row_and = bin(shifted & static[slide_row + row_shift])  
                        overlap += row_and.count("1")               
                    max_overlap = max(max_overlap, overlap)
        return max_overlap
EOF
output = divmod(int(input().strip()), int(input().strip()))
print(output[0])
print(output[1])
print(output)
EOF
class Solution(object):
    def encode(self, num):
        result = []
        while num:
            result.append('0' if num%2 else '1')
            num = (num-1)//2
        return "".join(reversed(result))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def splitArraySameAverage(self, A):
        def n_sum_target(n, tgt, j):    
            if (n, tgt, j) in invalid:  
                return False
            if n == 0:                  
                return tgt == 0         
            for i in range(j, len(C)):
                if C[i] > tgt:          
                    break
                if n_sum_target(n - 1, tgt - C[i], i + 1):  
                    return True
            invalid.add((n, tgt, j))
            return False
        n, sum_A = len(A), sum(A)
        invalid = set()                 
        C = sorted(A)                   
        for len_B in range(1, (n // 2) + 1):  
            target = sum_A * len_B / float(n)
            if target != int(target):  
                continue
            if n_sum_target(len_B, target, 0):
                return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def reorderLogFiles(self, logs):
        letters, numbers = [], []
        digits = {str(i) for i in range(10)}
        for log in logs:
            space = log.find(" ")
            first = log[space + 1]
            if first in digits:
                numbers.append(log)
            else:
                letters.append((log[space + 1:] + log[:space], log))
        letters.sort()  
        return [log for _, log in letters] + numbers
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def removeElements(self, head, val):
        dummy = prev = ListNode(None)   
        dummy.next = head
        while head:
            if head.val == val:
                prev.next, head.next, head = head.next, None, head.next
            else:
                prev, head = head, head.next
        return dummy.next
EOF
if __name__ == "__main__":
    t = int(input().strip())
    for _ in range(t):
        num_cnt = int(input().strip())
        deq = deque(list(map(int, input().strip().split(' '))))
        prev = max(deq[0], deq[-1])
        while deq:
            if prev >= deq[0] and prev >= deq[-1]:
                if deq[0] >= deq[-1]:
                    prev = deq.popleft()
                else:
                    prev = deq.pop()
            else:
                break
        if len(deq) == 0:
            print('Yes')
        else:
            print('No')
EOF
def is_happy(b):
    if b[0] != b[1] or b[-1] != b[-2]:
        return False
    for ind in range(1, len(b)-1):
        if b[ind] != b[ind-1] and b[ind] != b[ind+1]:
            return False
    return True
def happyLadybugs(b):
    cnt = Counter(b)
    print("cnt = {}".format(cnt))
    singles = list(filter(lambda x: x[0] != '_' and x[1] == 1, cnt.items()))
    empty = b.count('_')
    print("singles = {}".format(singles))
    print("empty = {}".format(empty))
    if len(singles) == 0 and empty > 0:
        return 'YES'
    elif len(b) > 2 and is_happy(b):
        return 'YES'
    else:
        return 'NO'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    g = int(input())
    for g_itr in range(g):
        n = int(input())
        b = input()
        result = happyLadybugs(b)
        fptr.write(result + '\n')
    fptr.close()
EOF
class Solution(object):
    def judgeSquareSum(self, c):
        for a in xrange(int(math.sqrt(c))+1):
            b = int(math.sqrt(c-a**2))
            if a**2 + b**2 == c:
                return True
        return False
EOF
class Solution(object):
    def minRemoveToMakeValid(self, s):
        result = list(s)
        count = 0
        for i, v in enumerate(result):
            if v == '(':
                count += 1
            elif v == ')':
                if count:
                    count -= 1
                else:
                    result[i] = ""
        if count:
            for i in reversed(xrange(len(result))):
                if result[i] == '(':
                    result[i] = ""
                    count -= 1
                    if not count:
                        break
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countBinarySubstrings(self, s):
        seq, prev_seq = 1, 0                    
        count = 0
        for i, c in enumerate(s[1:], 1):        
            if c != s[i - 1]:                   
                count += min(seq, prev_seq)     
                seq, prev_seq = 1, seq
            else:
                seq += 1
        return count + min(seq, prev_seq)       
EOF
def twoStrings(s1, s2):
    if len(set(s1) & set(s2)) > 0:
        return 'YES'
    else:
        return 'NO'
q = int(input().strip())
for a0 in range(q):
    s1 = input().strip()
    s2 = input().strip()
    result = twoStrings(s1, s2)
    print(result)
EOF
class Solution(object):
    def hasAlternatingBits(self, n):
        n, curr = divmod(n, 2)
        while n > 0:
            if curr == n % 2:
                return False
            n, curr = divmod(n, 2)
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def search(self, nums, target):
        return self.binary(nums, 0, len(nums)-1, target)
    def binary(self, nums, left, right, target):
        if left > right:
            return False
        mid = (left + right) // 2
        if nums[mid] == target:
            return True
        if nums[left] < nums[mid]:          
            if target < nums[mid] and target >= nums[left]:     
                return self.binary(nums, left, mid-1, target)   
            return self.binary(nums, mid+1, right, target)      
        if nums[mid] < nums[right]:         
            if target > nums[mid] and target <= nums[right]:    
                return self.binary(nums, mid+1, right, target)  
            return self.binary(nums, left, mid-1, target)       
        if nums[left] == nums[mid] and nums[mid] != nums[right]:    
            return self.binary(nums, mid+1, right, target)          
        if nums[right] == nums[mid] and nums[mid] != nums[left]:    
            return self.binary(nums, left, mid-1, target)           
        if self.binary(nums, left, mid-1, target):      
            return True
        return self.binary(nums, mid+1, right, target)
EOF
pre = [
        [[8, 1, 6], [3, 5, 7], [4, 9, 2]],
        [[6, 1, 8], [7, 5, 3], [2, 9, 4]],
        [[4, 9, 2], [3, 5, 7], [8, 1, 6]],
        [[2, 9, 4], [7, 5, 3], [6, 1, 8]], 
        [[8, 3, 4], [1, 5, 9], [6, 7, 2]],
        [[4, 3, 8], [9, 5, 1], [2, 7, 6]], 
        [[6, 7, 2], [1, 5, 9], [8, 3, 4]], 
        [[2, 7, 6], [9, 5, 1], [4, 3, 8]],
        ]
def formingMagicSquare(s):
    results = []
    for p in pre:
        total = 0
        for p_row, s_row in zip(p, s):
            for i, j in zip(p_row, s_row):
                if not i == j:
                    total += max([i, j]) - min([i, j])
        results.append(total)
    return min(results)
if __name__ == "__main__":
    s = []
    for s_i in range(3):
        s_t = [int(s_temp) for s_temp in input().strip().split(' ')]
        s.append(s_t)
    result = formingMagicSquare(s)
    print(result)
EOF
def permutationEquation(p):
    output = []
    for num in range(1, max(p)+1):
        output.append(p.index(p.index(num)+1)+1)
    return output
if __name__ == "__main__":
    n = int(input().strip())
    p = list(map(int, input().strip().split(' ')))
    result = permutationEquation(p)
    print ("\n".join(map(str, result)))
EOF
class TrieNode(object):
    def __init__(self):
        self.is_string = False
        self.leaves = {}
    def insert(self, word):
        cur = self
        for c in word:
            if not c in cur.leaves:
                cur.leaves[c] = TrieNode()
            cur = cur.leaves[c]
        cur.is_string = True
class Solution(object):
    def findWords(self, board, words):
        visited = [[False for j in xrange(len(board[0]))] for i in xrange(len(board))]
        result = {}
        trie = TrieNode()
        for word in words:
            trie.insert(word)
        for i in xrange(len(board)):
            for j in xrange(len(board[0])):
                self.findWordsRecu(board, trie, 0, i, j, visited, [], result)
        return result.keys()
    def findWordsRecu(self, board, trie, cur, i, j, visited, cur_word, result):
        if not trie or i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or visited[i][j]:
            return
        if board[i][j] not in trie.leaves:
            return
        cur_word.append(board[i][j])
        next_node = trie.leaves[board[i][j]]
        if next_node.is_string:
            result["".join(cur_word)] = True
        visited[i][j] = True
        self.findWordsRecu(board, next_node, cur + 1, i + 1, j, visited, cur_word, result)
        self.findWordsRecu(board, next_node, cur + 1, i - 1, j, visited, cur_word, result)
        self.findWordsRecu(board, next_node, cur + 1, i, j + 1, visited, cur_word, result)
        self.findWordsRecu(board, next_node, cur + 1, i, j - 1, visited, cur_word, result)
        visited[i][j] = False
        cur_word.pop()
EOF
def powerSum(X, N, num):
    value = X - num**N
    if value < 0:
        return 0
    elif value == 0:
        return 1
    else:
        return powerSum(value, N, num+1) + powerSum(X, N, num+1)
if __name__ == "__main__":
    X = int(input().strip())
    N = int(input().strip())
    result = powerSum(X, N, 1)
    print(result)
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def insertNodeAtTail(head, data):
    if not head:
        return SinglyLinkedListNode(data)
    cur = head
    while cur.next:
        cur = cur.next
    cur.next = SinglyLinkedListNode(data)
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()
    for i in range(llist_count):
        llist_item = int(input())
        llist_head = insertNodeAtTail(llist.head, llist_item)
        llist.head = llist_head
    print_singly_linked_list(llist.head, '\n', fptr)
    fptr.write('\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxVacationDays(self, flights, days):
        cities = len(flights)
        weeks = len(days[0])
        if not cities or not weeks:
            return 0
        prev_week_max_days = [0 for _ in range(cities)]
        for week in range(weeks - 1, -1, -1):
            this_week_max_days = [0 for _ in range(cities)]
            for start in range(cities):
                max_vacation = days[start][week] + prev_week_max_days[start]  
                for end in range(cities):
                    if flights[start][end]:  
                        max_vacation = max(max_vacation, days[end][week] + prev_week_max_days[end])
                this_week_max_days[start] = max_vacation
            prev_week_max_days = this_week_max_days
        return this_week_max_days[0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxChunksToSorted(self, arr):
        min_right = [float("inf") for _ in range(len(arr))]     
        for i in range(len(arr) - 2, -1, -1):
            min_right[i] = min(min_right[i + 1], arr[i + 1])
        partitions = 0
        partition_max = None
        for i, num in enumerate(arr):
            partition_max = num if partition_max is None else max(partition_max, num)
            if partition_max <= min_right[i]:
                partitions += 1
                partition_max = None
        return partitions
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def reverseList(self, head):
        reversed = None
        while head:
            next = head.next
            head.next = reversed
            reversed = head
            head = next
        return reversed
EOF
class Points(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    def __sub__(self, no):
        return Points(self.x - no.x,
                      self.y - no.y,
                      self.z - no.z)
    def dot(self, no):
        return self.x*no.x + self.y*no.y + self.z*no.z
    def cross(self, no):
        return Points(self.y*no.z - self.z*no.y,
                      self.z*no.x - self.x*no.z,
                      self.y*no.x - self.x*no.y)
    def absolute(self):
        return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)
EOF
def do_day(arr):
    prev = arr[0]
    res = [arr[0]]
    for el in arr[1:]:
        temp = el
        if el <= prev:
            res.append(el)
        prev = temp
    return res
def poisonousPlants_brute(arr):
    res = 0
    arr_prev = []
    arr_cur = arr
    while arr_cur != arr_prev:
        arr_cur, arr_prev = do_day(list(arr_cur)), arr_cur
        res += 1
    return res - 1
def poisonousPlants(arr):
    stack = [0]
    days = [0] * len(arr)
    pivot = arr[0]
    res = 0
    for ind in range(1, len(arr)):
        if arr[ind] > arr[ind - 1]:
            days[ind] = 1
        pivot = min(pivot, arr[ind])
        while stack and arr[stack[-1]] >= arr[ind]:
            if arr[ind] > pivot:
                days[ind] = max(days[ind], days[stack[-1]] + 1)
            stack.pop()
        res = max(res, days[ind])
        stack.append(ind)
    return res
if __name__ == "__main__":
    n = int(input().strip())
    p = list(map(int, input().strip().split(' ')))
    result = poisonousPlants(p)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Node:
    def __init__(self):
        self.children = {}  
        self.word = None
class Solution(object):
    def findWords(self, board, words):
        root = Node()
        for word in words:      
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = Node()
                node = node.children[c]
            node.word = word    
        found = []
        for r in range(len(board)):
            for c in range(len(board[0])):
                self.search(board, root, r, c, found)
        return found
    def search(self, board, node, r, c, found):     
        if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]):
            return
        letter = board[r][c]
        if letter not in node.children:
            return
        node = node.children[letter]
        if node.word:
            found.append(node.word)
            node.word = None    
        board[r][c] = '*'       
        self.search(board, node, r+1, c, found)
        self.search(board, node, r-1, c, found)
        self.search(board, node, r, c+1, found)
        self.search(board, node, r, c-1, found)
        board[r][c] = letter    
EOF
i = 4
d = 4.0
s = 'HackerRank'
i1 = int(input())
d1 = float(input())
s1 = input()
print(i+i1)
print(d+d1)
print(s+s1)
EOF
def reverse_array(arr):
    n = len(arr)
    for i in range(n-1, -1, -1):
        print(str(arr[i]), end = ' ')
n = int(input().strip())
arr = [int(arr_temp) for arr_temp in input().strip().split(' ')]
reverse_array(arr)
EOF
def hourglassSum(arr):
    maxx = -81
    for i in range(4):
        for j in range(4):
            total = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2];
            if total > maxx:
                maxx = total
    return maxx 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))
    result = hourglassSum(arr)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minWindow(self, S, T):
        next_in_s = [None for _ in range(len(S))]   
        next_by_letter = [-1 for _ in range(26)]    
        for i in range(len(S) - 1, -1, -1):
            next_in_s[i] = next_by_letter[:]
            next_by_letter[ord(S[i]) - ord("a")] = i
        matches = [[i, i] for i, c in enumerate(S) if c == T[0]]
        if not matches:
            return ""
        for i, c in enumerate(T[1:], 1):        
            new_matches = []
            for s_start, s_last in matches:     
                s_next = next_in_s[s_last][ord(c) - ord("a")]   
                if s_next != -1:
                    new_matches.append([s_start, s_next])       
            if not new_matches:
                return ""
            matches = new_matches
        start, end = min(matches, key = lambda i, j: j - i)
        return S[start:end + 1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def movesToChessboard(self, board):
        n = len(board)
        rows = {tuple(row) for row in board}    
        cols = set(zip(*board))                 
        moves = 0
        for patterns in [rows, cols]:           
            if len(patterns) != 2:              
                return -1
            p1, p2 = list(patterns)
            zero_p1, zero_p2 = sum(x == 0 for x in p1), sum(x == 0 for x in p2)
            if abs(zero_p1 - zero_p2) != n % 2 or not all(x ^ y for x, y in zip(p1, p2)):   
                return -1
            p = p1 if zero_p1 > zero_p2 else p2 
            p_moves = sum(x != y for x, y in zip(p, [0, 1] * ((n + 1) // 2)))
            if n % 2 == 0:                      
                p_moves = min(p_moves, sum(x != y for x, y in zip(p, [1, 0] * ((n + 1) // 2))))
            moves += p_moves // 2               
        return moves
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isMatch(self, s, p):
        matched = [[False for _ in range(len(p)+1)] for _  in range(len(s)+1)]
        matched[0][0] = True        
        for i in range(len(s)+1):
            for j in range(1, len(p)+1):
                pattern = p[j-1]
                if pattern == '.':      
                    matched[i][j] = (i != 0 and matched[i-1][j-1])
                elif pattern == '*':    
                    star = p[j-2]       
                    matched[i][j] = matched[i][j-2] or (i > 0 and matched[i-1][j] and (star == s[i-1] or star == '.'))
                else:                   
                    matched[i][j] = (i != 0 and matched[i-1][j-1] and s[i-1] == pattern)
        return matched[-1][-1]
EOF
n = int(input().strip())
for i in range(0, n):
    print(" " * (n-(i+1)) + "
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def isValidBST(self, root):
        self.correct = True
        self.prev = float('-inf')
        self.inorder(root)
        return self.correct
    def inorder(self, node):
        if not node or not self.correct:    
            return
        self.inorder(node.left)
        if node.val <= self.prev:
            self.correct = False
            return          
        self.prev = node.val
        self.inorder(node.right)
class Solution2(object):
    def isValidBST(self, root):
        return self.valid(root, float('-inf'), float('inf'))
    def valid(self, node, lower, upper):    
        if not node:
            return True
        if node.val <= lower or node.val >= upper:  
            return False
        return self.valid(node.left, lower, node.val) and self.valid(node.right, node.val, upper)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findTargetSumWays(self, nums, S):
        sums = defaultdict(int)
        sums[0] = 1
        running = nums[:]       
        for i in range(len(nums) - 2, -1, -1):
            running[i] += running[i + 1]
        for i, num in enumerate(nums):
            new_sums = defaultdict(int)
            for old_sum in sums:
                if S <= old_sum + running[i]:   
                    new_sums[old_sum + num] += sums[old_sum]
                if S >= old_sum - running[i]:
                    new_sums[old_sum - num] += sums[old_sum]
            sums = new_sums
        if S not in sums:
            return 0
        return sums[S]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxSubArray(self, nums):
        overall_max = float('-inf')
        max_ending_here = 0
        for num in nums:
            if max_ending_here > 0:
                max_ending_here += num
            else:
                max_ending_here = num
            overall_max = max(overall_max, max_ending_here)
        return overall_max
EOF
def wrap(string, max_width):
    return textwrap.fill(string, max_width)
if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
EOF
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    print(*arr[::-1])
EOF
	
class Person:
    def __init__(self,initialAge):
        if initialAge >= 0:
            self.age = initialAge
        else:
            self.age = 0
            print("Age is not valid, setting age to 0.")
    def amIOld(self):
        if self.age < 13:
            print("You are young.")
        elif self.age in range(13,18):
            print("You are a teenager.")
        else:
            print("You are old.")
    def yearPasses(self):
        self.age += 1
t = int(input())
for i in range(0, t):
    age = int(input())         
    p = Person(age)  
    p.amIOld()
    for j in range(0, 3):
        p.yearPasses()       
    p.amIOld()
    print("")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def sumSubarrayMins(self, A):
        A = [float("-inf")] + A + [float("-inf")]
        result = 0
        stack = []
        for i, num in enumerate(A):
            while stack and num < A[stack[-1]]:
                j = stack.pop()
                result += A[j] * (j - stack[-1]) * (i - j)  
            stack.append(i)
        return result % (10 ** 9 + 7)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maximumGap(self, nums):
        if len(nums) < 2:
            return 0
        lower = min(nums)
        difference = max(nums) - lower
        gaps = len(nums) - 1        
        if difference == 0:         
            return 0
        width = difference // gaps  
        if width == 0:
            width = 1
        nb_buckets = 1 + difference // width    
        buckets = [[None, None] for _ in range(nb_buckets)] 
        for num in nums:
            bucket = (num - lower) // width
            buckets[bucket][0] = min(buckets[bucket][0], num) if buckets[bucket][0] != None else num
            buckets[bucket][1] = max(buckets[bucket][1], num) if buckets[bucket][1] != None else num
        last_used_bucket = 0
        max_gap = difference // gaps  
        for i in range(nb_buckets - 1):
            if not buckets[i][0] and buckets[i + 1][0]:
                max_gap = max(max_gap, buckets[i + 1][0] - buckets[last_used_bucket][1])
            elif buckets[i][0]:
                last_used_bucket = i
        return max_gap
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestUnivaluePath(self, root):
        self.longest = 0
        def helper(node):
            if not node:
                return 0, 0
            max_left = max(helper(node.left))       
            max_right = max(helper(node.right))
            left = 1 + max_left if node.left and node.left.val == node.val else 0
            right = 1 + max_right if node.right and node.right.val == node.val else 0
            self.longest = max(self.longest, left + right)
            return left, right
        helper(root)
        return self.longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def asteroidCollision(self, asteroids):
        stack = []
        for asteroid in asteroids:
            while stack and stack[-1] > 0 > asteroid:  
                if stack[-1] == -asteroid:      
                    stack.pop()
                    break
                elif stack[-1] > -asteroid:     
                    break
                else:                           
                    stack.pop()
            else:  
                stack.append(asteroid)
        return stack
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def binaryTreePaths(self, root):
        def helper(node, partial):          
            if not node:
                return
            partial.append(str(node.val))
            if not node.left and not node.right:
                paths.append("->".join(partial))
                return
            helper(node.left, partial[:])
            helper(node.right, partial)
        paths = []
        helper(root, [])
        return paths
EOF
def compute_happiness(arr, set_a, set_b):
    happiness_count = 0
    for i in arr:
            if i in set_a:
                happiness_count += 1
            elif i in set_b:
                happiness_count -= 1
    print( happiness_count)
n, m = input().split()
nos = list(map(int, input().split()))
set_a = set(map(int,input().split()))
set_b = set(map(int,input().split()))
compute_happiness(nos,set_a,set_b)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findPairs(self, nums, k):
        if k < 0:
            return 0
        freq = Counter(nums)
        pairs = 0
        for num in freq:
            if k == 0:
                if freq[num] > 1:
                    pairs += 1
            else:
                if num + k in freq:
                    pairs += 1
        return pairs
EOF
np.set_printoptions(sign=' ')
m, n = map(int, input().split(' '))
print(np.eye(m, n))
EOF
i = 4
d = 4.0
s = 'HackerRank '
print( int(input()) + i)
print( float(input()) + d)
print( s + input())
EOF
for i in range(1,int(input())): 
    print(10**i//9 * i)
EOF
class Solution(object):
    def closestKValues(self, root, target, k):
        def nextNode(stack, child1, child2):
            if stack:
                if child2(stack):
                    stack.append(child2(stack))
                    while child1(stack):
                        stack.append(child1(stack))
                else:
                    child = stack.pop()
                    while stack and child is child2(stack):
                        child = stack.pop()
        backward = lambda stack: stack[-1].left
        forward = lambda stack: stack[-1].right
        stack = []
        while root:
            stack.append(root)
            root = root.left if target < root.val else root.right
        dist = lambda node: abs(node.val - target)
        forward_stack = stack[:stack.index(min(stack, key=dist))+1]
        backward_stack = list(forward_stack)
        nextNode(backward_stack, backward, forward)
        result = []
        for _ in xrange(k):
            if forward_stack and \
                (not backward_stack or dist(forward_stack[-1]) < dist(backward_stack[-1])):
                result.append(forward_stack[-1].val)
                nextNode(forward_stack, forward, backward)
            elif backward_stack and \
                (not forward_stack or dist(backward_stack[-1]) <= dist(forward_stack[-1])):
                result.append(backward_stack[-1].val)
                nextNode(backward_stack, backward, forward)
        return result
class Solution2(object):
    def closestKValues(self, root, target, k):
        class BSTIterator:
            def __init__(self, stack, child1, child2):
                self.stack = list(stack)
                self.cur = self.stack.pop()
                self.child1 = child1
                self.child2 = child2
            def next(self):
                node = None
                if self.cur and self.child1(self.cur):
                    self.stack.append(self.cur)
                    node = self.child1(self.cur)
                    while self.child2(node):
                        self.stack.append(node)
                        node = self.child2(node)
                elif self.stack:
                    prev = self.cur
                    node = self.stack.pop()
                    while node:
                        if self.child2(node) is prev:
                            break
                        else:
                            prev = node
                            node = self.stack.pop() if self.stack else None
                self.cur = node
                return node
        stack = []
        while root:
            stack.append(root)
            root = root.left if target < root.val else root.right
        dist = lambda node: abs(node.val - target) if node else float("inf")
        stack = stack[:stack.index(min(stack, key=dist))+1]
        backward = lambda node: node.left
        forward = lambda node: node.right
        smaller_it, larger_it = BSTIterator(stack, backward, forward), BSTIterator(stack, forward, backward)
        smaller_node, larger_node = smaller_it.next(), larger_it.next()
        result = [stack[-1].val]
        for _ in xrange(k - 1):
            if dist(smaller_node) < dist(larger_node):
                result.append(smaller_node.val)
                smaller_node = smaller_it.next()
            else:
                result.append(larger_node.val)
                larger_node = larger_it.next()
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMaxLength(self, nums):
        max_len = 0
        balance = 0         
        balances = {0: -1}  
        for i, num in enumerate(nums):
            if num == 1:
                balance += 1
            else:
                balance -= 1
            if balance in balances:
                max_len = max(max_len, i - balances[balance])
            else:
                balances[balance] = i
        return max_len
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def subarraysWithKDistinct(self, A, K):
        def at_most_k(distinct):
            count = Counter()
            subarrays = 0
            start = 0
            for end, num in enumerate(A):
                if count[num] == 0:
                    distinct -= 1
                count[num] += 1
                while distinct < 0:     
                    count[A[start]] -= 1
                    if count[A[start]] == 0:
                        distinct += 1
                    start += 1          
                subarrays += end - start + 1
            return subarrays
        return at_most_k(K) - at_most_k(K - 1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def eventualSafeNodes(self, graph):
        outgoing = [set(nbors) for nbors in graph]      
        incoming = [[] for _ in range(len(graph))]      
        for node, nbors in enumerate(graph):
            for nbor in nbors:
                incoming[nbor].append(node)
        safe = [node for node, nbors in enumerate(outgoing) if not nbors]   
        for safe_node in safe:                          
            nbors = incoming[safe_node]
            for nbor in nbors:
                outgoing[nbor].remove(safe_node)        
                if not outgoing[nbor]:                  
                    safe.append(nbor)
        return [node for node, nbors in enumerate(outgoing) if not nbors]   
EOF
def hourglassSum(arr):
    maxx = -81
    for i in range(4):
        for j in range(4):
            total = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2]
            if total > maxx:
                maxx = total
    return maxx 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))
    result = hourglassSum(arr)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def majorityElement(self, nums):
        count, candidate = 0, None
        for i, num in enumerate(nums):
            if count == 0:
                candidate = num
            if candidate == num:
                count += 1
            else:
                count -= 1
        return candidate
EOF
def solve(arr, money):
    cost_map = {}
    for i, cost in enumerate(arr):
        johnny = money - cost
        if johnny in cost_map.keys():
            print("{} {}".format(cost_map[johnny]+1, i+1))
        else:
            cost_map[cost] = i
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        money = int(input().strip())
        n = int(input().strip())
        arr = list(map(int, input().strip().split(' ')))
        solve(arr, money)
EOF
class Solution(object):
    def findPermutation(self, s):
        result = []
        for i in xrange(len(s)+1):
            if i == len(s) or s[i] == 'I':
                result += range(i+1, len(result), -1)
        return result
EOF
def check_interval(i, j):
    res = 3
    for inter in range(i, j + 1):
        res = min(res, width[inter])
    return res
n,t = input().strip().split(' ')
n,t = [int(n),int(t)]
width = [int(width_temp) for width_temp in input().strip().split(' ')]
for a0 in range(t):
    i,j = input().strip().split(' ')
    i,j = [int(i),int(j)]
    print(check_interval(i, j))
EOF
class Solution(object):
    def shortestDistance(self, grid):
        def bfs(grid, dists, cnts, x, y):
            dist, m, n = 0, len(grid), len(grid[0])
            visited = [[False for _ in xrange(n)] for _ in xrange(m)]
            pre_level = [(x, y)]
            visited[x][y] = True
            while pre_level:
                dist += 1
                cur_level = []
                for i, j in pre_level:
                    for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        I, J = i+dir[0], j+dir[1]
                        if 0 <= I < m and 0 <= J < n and grid[I][J] == 0 and not visited[I][J]:
                            cnts[I][J] += 1
                            dists[I][J] += dist
                            cur_level.append((I, J))
                            visited[I][J] = True
                pre_level = cur_level
        m, n, cnt = len(grid),  len(grid[0]), 0
        dists = [[0 for _ in xrange(n)] for _ in xrange(m)]
        cnts = [[0 for _ in xrange(n)] for _ in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                if grid[i][j] == 1:
                    cnt += 1
                    bfs(grid, dists, cnts, i, j)
        shortest = float("inf")
        for i in xrange(m):
            for j in xrange(n):
                if dists[i][j] < shortest and cnts[i][j] == cnt:
                    shortest = dists[i][j]
        return shortest if shortest != float("inf") else -1
EOF
def getMoneySpent(keyboards, drives, s):
    res = -1
    variants = list(filter(lambda x: sum(x) <= s, list(product(keyboards, drives))))
    if variants:
        res = sum(max(variants, key = sum))
    return res
s,n,m = input().strip().split(' ')
s,n,m = [int(s),int(n),int(m)]
keyboards = list(map(int, input().strip().split(' ')))
drives = list(map(int, input().strip().split(' ')))
moneySpent = getMoneySpent(keyboards, drives, s)
print(moneySpent)
EOF
def cutTheSticks(arr):
    out = []
    while arr:
        out.append(len(arr))
        arr_min = min(arr)
        arr = list(map(lambda x: x - arr_min, arr))
        arr = list(filter(lambda x: x > 0, arr))
    return out
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = cutTheSticks(arr)
    print ("\n".join(map(str, result)))
EOF
def pairs(k, arr):
    res = 0
    memo = dict()
    for el in arr:
        if el-k in memo:
            res += 1
        if el+k in memo:
            res += 1
        memo[el] = True
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    arr = list(map(int, input().rstrip().split()))
    result = pairs(k, arr)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
class Solution(object):
    def confusingNumberII(self, N):
        lookup = {"0":"0", "1":"1", "6":"9", "8":"8", "9":"6"}
        centers = {"0":"0", "1":"1", "8":"8"}
        def totalCount(N):   
            s = str(N)
            total = 0 
            p = len(lookup)**(len(s)-1)
            for i in xrange(len(s)):
                if i+1 == len(s):
                    for c in lookup.iterkeys():
                        total += int(c <= s[i])
                    continue
                smaller = 0
                for c in lookup.iterkeys():
                    smaller += int(c < s[i])
                total += smaller * p
                if s[i] not in lookup:
                    break
                p //= len(lookup)
            return total
        def validCountInLessLength(N):
            s = str(N)
            valid = 0
            total = len(centers)
            for i in xrange(1, len(s), 2):
                if i == 1:
                    valid += total
                else:
                    valid += total * (len(lookup)-1)
                    total *= len(lookup)
            total = 1
            for i in xrange(2, len(s), 2):
                valid += total * (len(lookup)-1)
                total *= len(lookup)
            return valid
        def validCountInFullLength(N):
            s = str(N)
            half_s = s[:(len(s)+1)//2]
            total = 0
            p =  len(lookup)**(len(half_s)-2) * len(centers) if (len(s) % 2) else len(lookup)**(len(half_s)-1)
            choices = centers if (len(s) % 2) else lookup
            for i in xrange(len(half_s)):
                if i+1 == len(half_s):
                    for c in choices.iterkeys():
                        if c == '0' and i == 0:
                            continue
                        total += int(c < half_s[i])
                    if half_s[i] not in choices:
                        break
                    tmp = list(half_s)
                    for i in reversed(xrange(len(half_s)-(len(s) % 2))):
                        tmp.append(lookup[half_s[i]])
                    if int("".join(tmp)) <= N:
                        total += 1
                    continue
                smaller = 0
                for c in lookup.iterkeys():
                    if c == '0' and i == 0:
                        continue
                    smaller += int(c < half_s[i])
                total += smaller * p
                if half_s[i] not in lookup:
                    break
                p //= len(lookup)
            return total
        return totalCount(N) - validCountInLessLength(N) - validCountInFullLength(N)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkEqualTree(self, root):
        def make_sum(node):
            if not node:
                return 0
            node.val += make_sum(node.left) + make_sum(node.right)
            return node.val
        tree_sum = make_sum(root)
        if tree_sum % 2 == 1:       
            return False
        def find_split(node):
            if not node:
                return False
            if node.left and node.left.val == tree_sum // 2:
                return True
            if node.right and node.right.val == tree_sum // 2:
                return True
            return find_split(node.left) or find_split(node.right)
        return find_split(root)
EOF
class Graph:
    def __init__(self, n):
        self.size = n
        self.vert = dict.fromkeys([n for n in range(n)])
        for node in self.vert.keys():
            self.vert[node] = {}
    def print_graph(self):
        print(self.vert)
    def add_edge(self, x, y, w):
        if not y in self.vert[x].keys() or self.vert[x][y] > w:
            self.vert[x][y] = w
            self.vert[y][x] = w
    def dijkstra(self, graph, start):
        path = [-1] * graph.size
        path[start] = 0
        visited = []
        next_to_visit = {start:0}
        while bool(next_to_visit):
            node = min(next_to_visit, key=next_to_visit.get)
            del next_to_visit[node]
            for child in graph.vert[node].keys():
                if not child in visited:
                    temp_path = path[node] + graph.vert[node][child]
                    if path[child] == -1 or path[child] > temp_path:
                        path[child] = temp_path
                        next_to_visit[child] = temp_path
            visited.append(node)
        del path[start]
        return path
t = int(input().strip())
for a0 in range(t):
    n,m = input().strip().split(' ')
    n,m = [int(n),int(m)]
    graph = Graph(n)
    for a1 in range(m):
        x,y,r = input().strip().split(' ')
        x,y,r = [int(x),int(y),int(r)]
        graph.add_edge(x - 1, y - 1, r)
    s = int(input().strip())
    result = graph.dijkstra(graph, s-1)
    print (" ".join(map(str, result)))
EOF
    time = s[:-2].split(":")
    if time_type == "AM":
        if time[0] == "12":
            time[0] = "00"
            return ":".join(time)
        return s[:-2]
    else:
        if time[0] == "12":
            return s[:-2]
        time = s[:-2].split(":")
        time[0] = str(int(time[0])+12)
        return ":".join(time)
if __name__ == '__main__':
    f = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = timeConversion(s)
    f.write(result + '\n')
    f.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reorganizeString(self, S):
        freq = Counter(S)
        if any(count > (len(S) + 1) // 2 for count in freq.values()):   
            return ""
        heap = [(-count, letter) for letter, count in freq.items()]     
        heapq.heapify(heap)
        result = []
        def add_letter(letter, neg_count):
            result.append(letter)
            neg_count += 1
            if neg_count != 0:                                          
                heapq.heappush(heap, (neg_count, letter))
        while heap:
            neg_count, letter = heapq.heappop(heap)
            if not result or result[-1] != letter:
                add_letter(letter, neg_count)
                continue
            if not heap:                                                
                return ""
            neg_count2, letter2 = heapq.heappop(heap)                   
            add_letter(letter2, neg_count2)
            heapq.heappush(heap, (neg_count, letter))                   
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def rob(self, root):
        return max(self.helper(root))
    def helper(self, node):
        if not node:
            return 0, 0
        left_with, left_without = self.helper(node.left)
        right_with, right_without = self.helper(node.right)
        max_with = node.val + left_without + right_without
        max_without = max(left_with, left_without) + max(right_with, right_without)
        return max_with, max_without
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxCoins(self, nums):
        n = len(nums)
        nums = [1] + nums + [1]
        max_coins = [[0 for _ in range(n + 2)] for _ in range(n + 1)]   
        for length in range(1, n + 1):
            for left in range(1, n + 2 - length):   
                right = left + length - 1
                for last in range(left, right + 1):
                    this_coins = nums[left - 1] * nums[last] * nums[right + 1]  
                    max_coins[length][left] = max(max_coins[length][left],
                        this_coins + max_coins[last - left][left] + max_coins[right - last][last + 1])
        return max_coins[-1][1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sortColors(self, nums):
        next_red, next_white = 0, 0
        for i in range(len(nums)):
            colour = nums[i]
            nums[i] = 2
            if colour < 2:
                nums[next_white] = 1
                next_white += 1
            if colour == 0:
                nums[next_red] = 0
                next_red += 1
EOF
def mutate_string(string, position, character):
    out = list(string)
    out[position] = character
    return "".join(out)
EOF
class Solution(object):
    def minCut(self, s):
        lookup = [[False for j in xrange(len(s))] for i in xrange(len(s))]
        mincut = [len(s) - 1 - i for i in xrange(len(s) + 1)]
        for i in reversed(xrange(len(s))):
            for j in xrange(i, len(s)):
                if s[i] == s[j]  and (j - i < 2 or lookup[i + 1][j - 1]):
                    lookup[i][j] = True
                    mincut[i] = min(mincut[i], mincut[j + 1] + 1)
        return mincut[0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countComponents(self, n, edges):
        parents = [i for i in range(n)]
        components = n
        def update_parent(node):
            while node != parents[node]:
                parents[node] = parents[parents[node]]  
                node = parents[parents[node]]           
            return node
        for a, b, in edges:
            a_parent = update_parent(a)
            b_parent = update_parent(b)
            if a_parent != b_parent:
                parents[a_parent] = b_parent
                components -= 1
        return components
EOF
def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bs.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError
def median(a_sorted, days):
    half = len(a_sorted)//2
    if days % 2:
        median = a_sorted[half]
    else:
        median = (a_sorted[half-1] + a_sorted[half])/2
    return float(median)
def activityNotifications(log, days):
    heap = sorted(log[:days])
    res = 0
    med = 0
    to_del = 0
    for ind in range(days, len(log)):
        med = median(heap, days)
        if float(log[ind]) >= 2*med:
            res += 1
        del heap[index(heap, log[to_del])]
        bs.insort(heap, log[ind])
        to_del += 1
    return res
if __name__ == "__main__":
    n, d = input().strip().split(' ')
    n, d = [int(n), int(d)]
    expenditure = list(map(int, input().strip().split(' ')))
    result = activityNotifications(expenditure, d)
    print(result)
EOF
def how_many_games(p, d, m, s):
    res = 0
    while s > 0:
        res += 1
        s -= p
        p = max(p - d, m)
    if s != 0:
        res -= 1
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    p, d, m, s = [int(n) for n in input().split(" ")]
    answer = how_many_games(p, d, m, s)
    fptr.write(str(answer) + '\n')
    fptr.close()
EOF
class Solution(object):
    def isGoodArray(self, nums):
        def gcd(a, b):
            while b:
                a, b = b, a%b
            return a
        result = nums[0]
        for num in nums:
            result = gcd(result, num)
            if result == 1:
                break
        return result == 1
EOF
def list_len(head):
    node = head
    res = 0
    while node != None:
        res += 1
        node = node.next
    return res
def print_list(head):
    node = head
    while node != None:
        print("{}".format(node.data), end='')
        node = node.next
    print()
def InsertNth(head, data, position):
    if list_len(head) == 1 and head.data == 2:
        head = Node(data = data)
    elif position == 0:
        head = Node(data = data, next_node = head)
    else:
        node = head
        for _ in range(position - 1):
            node = node.next
        node.next = Node(data, next_node = node.next)
    return head
EOF
class Solution(object):
    def removeCoveredIntervals(self, intervals):
        intervals.sort(key=lambda x: [x[0], -x[1]])
        result, max_right = 0, 0
        for left, right in intervals:
            result += int(right > max_right)
            max_right = max(max_right, right)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def readBinaryWatch(self, num):
        if num == 0:
            return ["0:00"]
        bits_set = [[i] for i in range(10)]         
        for max_bit in range(10 - num + 1, 10):     
            new_bits_set = []
            for time in bits_set:
                for bit in range(time[-1] + 1, max_bit + 1):
                    new_bits_set.append(time + [bit])   
            bits_set = new_bits_set
        result = []
        for time in bits_set:
            hours, mins = 0, 0
            for bit in time:
                if bit >= 6:                        
                    hours += 1 << (bit - 6)
                else:                               
                    mins += 1 << bit
            if hours < 12 and mins < 60:            
                mins = str(mins)
                if len(mins) == 1:                  
                    mins = "0" + mins
                result.append(str(hours) + ":" + mins)
        return result
EOF
def timeConversion(s):
    if 'AM' in s:
        s_split = s.replace('AM', '').split(':')
        if s_split[0] == '12':
            s_split[0] = '00'
        res = s_split[0] + ':' + s_split[1] + ':' + s_split[2]
        return res
    else:
        s_split = s.replace('PM', '').split(':')
        if s_split[0] != '12':
            s_split[0] = str(int(s_split[0]) + 12)
        res = s_split[0] + ':' + s_split[1] + ':' + s_split[2]
        return res
s = input().strip()
result = timeConversion(s)
print(result)
EOF
def get_mask(n):
    return 1 << math.floor(math.log(n,2))
def counter_game(n):
    winner = 0
    while n != 1:
        mask = get_mask(n)
        if mask == n:
            n = n/2
        else:
            n -= mask
        winner ^= 1
    if winner == 1:
        return "Louise"
    elif winner == 0:
        return "Richard"
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = int(input().strip())
        result = counter_game(n)
        print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def mergeTrees(self, t1, t2):
        if t1 and t2:
            root = TreeNode(t1.val + t2.val)
            root.left = self.mergeTrees(t1.left, t2.left)
            root.right = self.mergeTrees(t1.right, t2.right)
            return root
        return t1 or t2
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findStrobogrammatic(self, n):
        if n <= 0:
            return ['']
        if n % 2 == 1:
            results = ['0', '1', '8']
        else:
            results = ['']
        strobo = {'0' : '0', '1' : '1', '8': '8', '6' : '9', '9' : '6'}
        for i in range(n//2):
            results = [c + r + strobo[c] for r in results for c in strobo]
        return [result for result in results if (result[0] != '0' or n == 1)]
EOF
class SkipNode(object):
    def __init__(self, level=0, num=None):
        self.num = num
        self.nexts = [None]*level
class Skiplist(object):
    P_NUMERATOR, P_DENOMINATOR = 1, 2  
    MAX_LEVEL = 32  
    def __init__(self):
        self.__head = SkipNode()
        self.__len = 0
    def search(self, target):
        return True if self.__find(target, self.__find_prev_nodes(target)) else False
    def add(self, num):
        node = SkipNode(self.__random_level(), num)
        if len(self.__head.nexts) < len(node.nexts): 
            self.__head.nexts.extend([None]*(len(node.nexts)-len(self.__head.nexts)))
        prevs = self.__find_prev_nodes(num)
        for i in xrange(len(node.nexts)):
            node.nexts[i] = prevs[i].nexts[i]
            prevs[i].nexts[i] = node
        self.__len += 1
    def erase(self, num):
        prevs = self.__find_prev_nodes(num)
        curr = self.__find(num, prevs)
        if not curr:
            return False
        self.__len -= 1   
        for i in reversed(xrange(len(curr.nexts))):
            prevs[i].nexts[i] = curr.nexts[i]
            if not self.__head.nexts[i]:
                self.__head.nexts.pop()
        return True
    def __find(self, num, prevs):
        if prevs:
            candidate = prevs[0].nexts[0]
            if candidate and candidate.num == num:
                return candidate
        return None
    def __find_prev_nodes(self, num):
        prevs = [None]*len(self.__head.nexts)
        curr = self.__head
        for i in reversed(xrange(len(self.__head.nexts))):
            while curr.nexts[i] and curr.nexts[i].num < num:
                curr = curr.nexts[i]
            prevs[i] = curr
        return prevs
    def __random_level(self):
        level = 1
        while random.randint(1, Skiplist.P_DENOMINATOR) <= Skiplist.P_NUMERATOR and \
              level < Skiplist.MAX_LEVEL:
            level += 1
        return level
    def __len__(self):
        return self.__len
    def __str__(self):
        result = []
        for i in reversed(xrange(len(self.__head.nexts))):
            result.append([])
            curr = self.__head.nexts[i]
            while curr:
                result[-1].append(str(curr.num))
                curr = curr.nexts[i]
        return "\n".join(map(lambda x: "->".join(x), result))
EOF
alphabet = string.ascii_lowercase
def is_pangram(s):
    return set(alphabet) < set(s.lower())
if __name__ == "__main__":
    if is_pangram(input().strip()):
        print("pangram")
    else:
        print("not pangram")
EOF
class Solution(object):
    def hasPath(self, maze, start, destination):
        def neighbors(maze, node):
            for i, j in [(-1, 0), (0, 1), (0, -1), (1, 0)]:
                x, y = node
                while 0 <= x + i < len(maze) and \
                      0 <= y + j < len(maze[0]) and \
                      not maze[x+i][y+j]:
                    x += i
                    y += j
                yield x, y
        start, destination = tuple(start), tuple(destination)
        queue = collections.deque([start])
        visited = set()
        while queue:
            node = queue.popleft()
            if node in visited: continue
            if node == destination:
                return True
            visited.add(node)
            for neighbor in neighbors(maze, node):
                queue.append(neighbor)
        return False
EOF
def insertionSort1(start, arr):
    probe = arr[start]
    for ind in range(start-1, -1, -1):
        if arr[ind] > probe:
            arr[ind+1] = arr[ind]
        else:
            arr[ind+1] = probe
            break
    if arr[0] > probe:
        arr[0] = probe
def insertionSort2(n, arr):
    for ind in range(1, len(arr)):
        insertionSort1(ind, arr)
        print(" ".join(map(str, arr)))
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    insertionSort2(n, arr)
EOF
int main() {
    int n; 
    int k; 
    scanf("%d %d",&n,&k);
    int *a = malloc(sizeof(int) * n);
    int *a_out = malloc(sizeof(int) * n);
    for(int a_i = 0; a_i < n; a_i++){
       scanf("%d",&a[a_i]);
    }
/*    
    for(int i = 0; i < k; i++){
        int buf = a[0];
        for (int j = 0; j < n-1; j++) {
            a[j] = a[j+1];
        }
        a[n-1] = buf;
    }
*/    
    for (int i = 0; i < n; i++) {
        a_out[(i+n-k)%n] = a[i];
    }
    for(int a_i = 0; a_i < n; a_i++){
       printf("%d ", a_out[a_i]);
    }   
    printf("\n");
    free(a);
    return 0;
}
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def xorGame(self, nums):
        return reduce(operator.xor, nums) == 0 or len(nums) % 2 == 0
EOF
class Solution(object):
    def countUnivalSubtrees(self, root):
        [is_uni, count] = self.isUnivalSubtrees(root, 0)
        return count
    def isUnivalSubtrees(self, root, count):
        if not root:
            return [True, count]
        [left, count] = self.isUnivalSubtrees(root.left, count)
        [right, count] = self.isUnivalSubtrees(root.right, count)
        if self.isSame(root, root.left, left) and \
           self.isSame(root, root.right, right):
                count += 1
                return [True, count]
        return [False, count]
    def isSame(self, root, child, is_uni):
        return not child or (is_uni and root.val == child.val)
EOF
thickness = int(input()) 
c = 'H'
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minStickers(self, stickers, target):
        target_set, remaining_target = set(target), set(target)
        char_to_word = defaultdict(set)  
        for sticker in stickers:
            cleaned = tuple(x for x in sticker if x in target_set)  
            sticker_set = set(cleaned)
            for c in sticker_set:
                char_to_word[c].add(cleaned)
            remaining_target -= sticker_set
        if remaining_target:
            return -1
        heap = [(0, len(target), list(target))]  
        while True:
            used_words, target_len, target_str = heapq.heappop(heap)  
            for sticker in char_to_word[target_str[0]]:  
                new_str = target_str[:]
                for ch in sticker:
                    if ch in new_str:
                        new_str.remove(ch)
                if not new_str:
                    return used_words + 1
                heapq.heappush(heap, (used_words + 1, len(new_str), new_str))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class NestedIterator(object):
    def __init__(self, nestedList):
        self.flat = []
        def flatten(nested):
            for n in nested:
                if n.isInteger():
                    self.flat.append(n.getInteger())
                else:
                    flatten(n.getList())
        flatten(nestedList)
        self.flat = self.flat[::-1]
    def next(self):
        return self.flat.pop()
    def hasNext(self):
        return bool(self.flat)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def insertionSortList(self, head):
        sorted_tail = dummy = ListNode(float('-inf'))   
        dummy.next = head
        while sorted_tail.next:
            node = sorted_tail.next
            if node.val >= sorted_tail.val:             
                sorted_tail = sorted_tail.next
                continue
            sorted_tail.next = sorted_tail.next.next    
            insertion = dummy
            while insertion.next.val <= node.val:
                insertion = insertion.next
            node.next = insertion.next                  
            insertion.next = node
        return dummy.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findDuplicates(self, nums):
        result = []
        for i in range(len(nums)):
            num = abs(nums[i])      
            if nums[num - 1] < 0:   
                result.append(num)
                continue
            nums[num - 1] = -nums[num - 1]  
        return result
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def removeDuplicates(head):
    cur = head
    while cur and cur.next:
        temp = cur.next
        while temp and cur.data == temp.data:
            temp = temp.next
        cur.next = temp
        cur = temp
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        llist1 = removeDuplicates(llist.head)
        print_singly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()
EOF
def equalizeArray(arr):
    cnt = Counter(arr)
    m_el_num = max(cnt.items(), key = lambda x: x[1])
    return len(arr) - m_el_num[1]
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = equalizeArray(arr)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minAbbreviation(self, target, dictionary):
        def abbr(target, num):
            word, count = [], 0  
            for w in target:
                if num & 1 == 1:  
                    if count:
                        word += str(count)
                        count = 0
                    word.append(w)
                else:  
                    count += 1
                num >>= 1  
            if count:
                word.append(str(count))
            return "".join(word)
        m = len(target)
        diffs = []  
        for word in dictionary:
            if len(word) != m:
                continue
            bits = 0
            for i, char in enumerate(word):
                if char != target[i]:  
                    bits += 2 ** i
            diffs.append(bits)
        if not diffs:
            return str(m)
        min_abbr = target
        for i in range(2 ** m):  
            if all(d & i for d in diffs):  
                abbr_i = abbr(target, i)
                if len(abbr_i) < len(min_abbr):
                    min_abbr = abbr_i
        return min_abbr
EOF
class Solution(object):
    def combinationSum2(self, candidates, target):
        result = []
        self.combinationSumRecu(sorted(candidates), result, 0, [], target)
        return result
    def combinationSumRecu(self, candidates, result, start, intermediate, target):
        if target == 0:
            result.append(list(intermediate))
        prev = 0
        while start < len(candidates) and candidates[start] <= target:
            if prev != candidates[start]:
                intermediate.append(candidates[start])
                self.combinationSumRecu(candidates, result, start + 1, intermediate, target - candidates[start])
                intermediate.pop()
                prev = candidates[start]
            start += 1
EOF
def maximumToys(prices, k):
    prices = sorted(prices)
    res = 0
    for el in prices:
        if k - el >= 0:
            k -= el
            res += 1
        else:
            break
    return res
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    prices = list(map(int, input().strip().split(' ')))
    result = maximumToys(prices, k)
    print(result)
EOF
def wrappers(wr, m):
    res = 0
    if wr//m > 0:
        res += wr//m + wrappers(wr//m + wr%m, m)
    return res
def chocolateFeast(n, c, m):
    return n//c + wrappers(n//c, m)
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n, c, m = input().strip().split(' ')
        n, c, m = [int(n), int(c), int(m)]
        result = chocolateFeast(n, c, m)
        print(result)
EOF
def findDigits(n):
    res = 0
    for dig in n:
        if dig != '0' and int(n) % int(dig) == 0:
            res += 1
    return res
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = input().strip()
        result = findDigits(n)
        print(result)
EOF
class Solution(object):
    def licenseKeyFormatting(self, S, K):
        result = []
        for i in reversed(xrange(len(S))):
            if S[i] == '-':
                continue
            if len(result) % (K + 1) == K:
                result += '-'
            result += S[i].upper()
        return "".join(reversed(result))
EOF
def quickSort(arr):
    left = []
    equal = []
    right = []
    pivot = arr[0]
    for el in arr:
        if el < pivot:
            left.append(el)
        elif el == pivot:
            equal.append(el)
        elif el > pivot:
            right.append(el)
    return left + equal + right
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = quickSort(arr)
    print (" ".join(map(str, result)))
EOF
class Solution(object):
    def cherryPickup(self, grid):
        dp = [[[float("-inf")]*(len(grid[0])+2) for _ in xrange(len(grid[0])+2)] for _ in xrange(2)]
        dp[0][1][len(grid[0])] = grid[0][0] + grid[0][len(grid[0])-1]
        for i in xrange(1, len(grid)):
            for j in xrange(1, len(grid[0])+1):
                for k in xrange(1, len(grid[0])+1):
                    dp[i%2][j][k] = max(dp[(i-1)%2][j+d1][k+d2] for d1 in xrange(-1, 2) for d2 in xrange(-1, 2)) + \
                                    ((grid[i][j-1]+grid[i][k-1]) if j != k else grid[i][j-1])
        return max(itertools.imap(max, *dp[(len(grid)-1)%2]))
class Solution2(object):
    def cherryPickup(self, grid):
        dp = [[[float("-inf")]*len(grid[0]) for _ in xrange(len(grid[0]))] for _ in xrange(2)]
        dp[0][0][len(grid[0])-1] = grid[0][0] + grid[0][len(grid[0])-1]
        for i in xrange(1, len(grid)):
            for j in xrange(len(grid[0])):
                for k in xrange(len(grid[0])):
                    dp[i%2][j][k] = max(dp[(i-1)%2][j+d1][k+d2] for d1 in xrange(-1, 2) for d2 in xrange(-1, 2)
                                        if 0 <= j+d1 < len(grid[0]) and 0 <= k+d2 < len(grid[0])) + \
                                    ((grid[i][j]+grid[i][k]) if j != k else grid[i][j])
        return max(itertools.imap(max, *dp[(len(grid)-1)%2]))
EOF
def theGreatXor(x):
    res = 0
    n_bin = bin(x).replace('0b', '')
    for ind, digit in enumerate(reversed(n_bin)):
        if digit == '0':
            res += pow(2, ind)
    return res
q = int(input().strip())
for a0 in range(q):
    x = int(input().strip())
    result = theGreatXor(x)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isPalindrome(self, x):
        s = str(x)
        left, right = 0, len(s)-1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -=1
        return True
EOF
def has_cycle(head):
    if not head:
        return False
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
def read4(buf):
    pass
class Solution:
    def read(self, buf, n):
        total_chars, last_chars = 0, 4
        while last_chars == 4 and total_chars < n:
            buf4 = [""] * 4     
            last_chars = min(read4(buf4), n - total_chars)
            buf[total_chars:total_chars+last_chars] = buf4
            total_chars += last_chars
        return total_chars
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def hasPath(self, maze, start, destination):
        queue = [start]
        dirns = ((1, 0), (-1, 0), (0, 1), (0, -1))
        visited = set()
        while queue:
            start_r, start_c = queue.pop()
            visited.add((start_r, start_c))
            for dr, dc in dirns:
                r, c = start_r, start_c     
                while 0 <= r + dr < len(maze) and 0 <= c + dc < len(maze[0]) and maze[r + dr][c + dc] == 0:
                    r += dr
                    c += dc
                if (r, c) not in visited:
                    if [r, c] == destination:
                        return True
                    queue.append((r, c))    
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findPeakElement(self, nums):
        left, right = 0, len(nums)-1
        while left < right - 1:     
            mid = (left + right) // 2
            if nums[mid] >= nums[mid+1] and nums[mid] >= nums[mid-1]:
                return mid
            if nums[mid+1] > nums[mid]:     
                left = mid + 1
            else:                           
                right = mid - 1
        if nums[left] >= nums[right]:
            return left
        return right
EOF
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
    def __repr__(self):
        if self:
            return "{} -> {}".format(self.val, self.next)
class Solution(object):
    def swapPairs(self, head):
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        while current.next and current.next.next:
            next_one, next_two, next_three = current.next, current.next.next, current.next.next.next
            current.next = next_two
            next_two.next = next_one
            next_one.next = next_three
            current = next_one
        return dummy.next
EOF
if __name__ == "__main__":
    out = list(re.split('[.,]', input()))
    print("\n".join(filter(lambda x: re.match('[0-9]+',x), out)))
EOF
def average(array):
    return sum(set(array))/len(set(array))
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def summaryRanges(self, nums):
        summary = []
        for num in nums:
            if not summary or num > summary[-1][1] + 1:
                summary.append([num, num])
            else:
                summary[-1][1] = num
        result = [str(i) if i == j else str(i) + '->' + str(j) for i, j in summary]
        return result
EOF
def alternatingCharacters(s):
    s = list(s)
    i = 0
    count = 0
    while i < len(s) - 1:
        if s[i] == s[i + 1]:
            del (s[i])
            count += 1
        else:
            i += 1
    return count
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = alternatingCharacters(s)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestDistance(self, maze, start, destination):
        rows, cols = len(maze), len(maze[0])
        distance = 0
        visited = set()
        dirns = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
        perps = {"u": ("l", "r"), "d": ("l", "r"), "l": ("u", "d"), "r": ("u", "d")}
        queue = [(start[0], start[1], d) for d in dirns]
        while queue:
            new_queue = []
            while queue:
                r, c, dirn = queue.pop()
                if ((r, c, dirn)) in visited:
                    continue
                visited.add((r, c, dirn))
                dr, dc = dirns[dirn]
                if 0 <= r + dr < rows and 0 <= c + dc < cols and maze[r + dr][c + dc] == 0:
                    new_queue.append((r + dr, c + dc, dirn))
                else:
                    if [r, c] == destination:
                        return distance
                    perp = perps[dirn]
                    for new_dirn in perp:
                        queue.append((r, c, new_dirn))
            distance += 1
            queue = new_queue
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMaximumXOR(self, nums):
        mask = 0        
        max_xor = 0     
        for bit in range(31, -1, -1):           
            mask |= (1 << bit)                  
            prefixes = {mask & num for num in nums}
            target = max_xor | (1 << bit)       
            for prefix in prefixes:             
                if prefix ^ target in prefixes:
                    max_xor = target
                    break
        return max_xor
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numBusesToDestination(self, routes, S, T):
        routes = [set(route) for route in routes]       
        stop_to_routes = defaultdict(set)               
        for route, stops in enumerate(routes):
            for stop in stops:
                stop_to_routes[stop].add(route)
        front, back = {S}, {T}                          
        visited = set()                                 
        buses = 0
        while front and back and not (front & back):    
            if len(front) < len(back):                  
                front, back = back, front
            buses += 1
            new_front = set()
            visited |= front                            
            for stop in front:
                for route in stop_to_routes[stop]:
                    new_front |= routes[route]          
            front = new_front - visited                 
        return buses if front & back else -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def splitArray(self, nums):
        n = len(nums)
        if n < 7:
            return False
        cumul = [nums[0]]           
        for num in nums[1:]:
            cumul.append(num + cumul[-1])
        for j in range(3, n - 3):   
            candidates = set()
            for i in range(1, j - 1):
                left_sum = cumul[i - 1]
                right_sum = cumul[j - 1] - cumul[i]
                if left_sum == right_sum:
                    candidates.add(left_sum)
            for k in range(j + 2, n - 1):
                left_sum = cumul[k - 1] - cumul[j]
                right_sum = cumul[n - 1] - cumul[k]
                if left_sum == right_sum and left_sum in candidates:
                    return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wiggleMaxLength(self, nums):
        if not nums:
            return 0
        max_length = 1
        prev = nums[0]
        direction = 0  
        for num in nums[1:]:
            if direction != -1 and num < prev:
                max_length += 1
                direction = -1
            elif direction != 1 and num > prev:
                max_length += 1
                direction = 1
            prev = num
        return max_length
EOF
d = defaultdict(list)
n,m = map(int,input().split())
for i in range(1,n+1):
    d[input()].append(i)
for i in range(m):
    temp = input()
    if temp in d:
        print(*d[temp]) 
    else: 
        print(-1)
EOF
class Node:
    def __init__(self, info): 
        self.info = info  
        self.left = None  
        self.right = None 
        self.level = None 
    def __str__(self):
        return str(self.info) 
class BinarySearchTree:
    def __init__(self): 
        self.root = None
    def create(self, val):  
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break
def levelOrder(root):
    level = [root]
    while level:
        temp = level.pop(0)
        print(temp.info,end=' ')
        if temp.left:
            level.append(temp.left)
        if temp.right:
            level.append(temp.right)
tree = BinarySearchTree()
t = int(input())
arr = list(map(int, input().split()))
for i in range(t):
    tree.create(arr[i])
levelOrder(tree.root)
EOF
def split_and_join(line):
    return(line.replace(' ', '-'))
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)
EOF
class Book(object, metaclass=ABCMeta):
    def __init__(self,title,author):
        self.title=title
        self.author=author   
        def display(): pass
class MyBook(Book):
    def __init__(self,title,author,price):
        super(Book, self).__init__()
        self.price = price 
    def display(self):
        print("Title: "+ title)
        print("Author: "+ author)
        print("Price:",price)
title=input()
author=input()
price=int(input())
new_novel=MyBook(title,author,price)
new_novel.display()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isPerfectSquare(self, num):
        left, right = 1, num
        while left <= right:
            mid = (left + right) // 2
            square = mid * mid
            if square == num:
                return True
            if square > num:
                right = mid - 1
            else:
                left = mid + 1
        return False
EOF
class Difference:
    def __init__(self, a):
        self.__elements = a
        self.maximumDifference = 0
    def computeDifference(self):
        self.maximumDifference = max(self.__elements) - min(self.__elements)
_ = input()
a = [int(e) for e in input().split(' ')]
d = Difference(a)
d.computeDifference()
print(d.maximumDifference)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def subarraysDivByK(self, A, K):
        result = 0
        running_sum = 0
        prefix_sums = defaultdict(int)
        prefix_sums[0] = 1
        for num in A:
            running_sum = (running_sum + num) % K
            result += prefix_sums[running_sum]
            prefix_sums[running_sum] += 1
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rand10(self):
        units = rand7() - 1
        sevens = rand7() - 1
        num = 7 * sevens + units    
        if num >= 40:               
            return self.rand10()
        return (num % 10) + 1       
EOF
m, english = input(), set(map(int, input().split()))
n, french = input(), set(map(int, input().split()))
print(len(english.symmetric_difference(french)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canAttendMeetings(self, intervals):
        intervals.sort(key=lambda x: x.start)
        for i, interval in enumerate(intervals[1:], 1):     
            if interval.start < intervals[i - 1].end:
                return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def islandPerimeter(self, grid):
        rows, cols = len(grid), len(grid[0])
        for row in grid:                
            row.append(0)
        grid.append([0] * (cols + 1))
        perimiter = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    perimiter += 4
                    if grid[r + 1][c] == 1:
                        perimiter -= 2
                    if grid[r][c + 1] == 1:
                        perimiter -= 2
        return perimiter
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isNumber(self, s):
        self.digits = {str(i) for i in range(10)}
        s = [c for c in s.strip().lower()]
        if not s:
            return False
        return self.is_int(s, True) or self.is_float(s) or self.is_sci(s)
    def is_int(self, s, signed):
        if len(s) == 0:
            return False
        if s[0] == '-' and signed:
            s = s[1:]
        elif s[0] == '+' and signed:
            s = s[1:]
        if len(s) == 0:
            return False
        for c in s:
            if c not in self.digits:
                return False
        return True
    def is_float(self, s):
        try:
            dot = s.index('.')
            before = s[:dot]
            after = s[dot+1:]
            if before and before[0] in '+-':
                before = before[1:]
            if before and not self.is_int(before, False):
                return False
            if after and not self.is_int(after, False):
                return False
            return before or after
        except:
            return False
    def is_sci(self, s):
        try:
            e = s.index('e')
            before = s[:e]
            after = s[e+1:]
            if not before or not after:
                return False
            if not self.is_int(before, True) and not self.is_float(before):
                return False
            return self.is_int(after, True)
        except:
            return False
EOF
def arrays(arr):
    return numpy.array(arr, float)[::-1]
arr = input().strip().split(' ')
result = arrays(arr)
print(result)
EOF
n = int(input().strip())
if n%2 != 0:
    print('Weird')
else:
    if n in range(2,6):
        print("Not Weird")
    elif n in range(6,21):
        print("Weird")
    elif n > 20:
        print("Not Weird")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def bestRotation(self, A):
        n = len(A)
        rotations = [0 for _ in range(n)]
        for i, num in enumerate(A):
            min_rot = (i + 1) % n               
            max_rot = (n - num + i + 1) % n     
            rotations[min_rot] += 1
            rotations[max_rot] -= 1
            if min_rot > max_rot:               
                rotations[0] += 1
        score, max_score, best_rotation = 0, 0, 0
        for i, r in enumerate(rotations):
            score += r
            if score > max_score:
                max_score = score
                best_rotation = i
        return best_rotation
EOF
class Solution(object):
    def totalFruit(self, tree):
        count = collections.defaultdict(int)
        result, i = 0, 0
        for j, v in enumerate(tree):
            count[v] += 1
            while len(count) > 2:
                count[tree[i]] -= 1
                if count[tree[i]] == 0:
                    del count[tree[i]]
                i += 1
            result = max(result, j-i+1)
        return result
EOF
if __name__ == "__main__":
    n = int(input().strip())
    english = set(map(int, input().strip().split(' ')))
    m = int(input().strip())
    french = set(map(int, input().strip().split(' ')))
    print(len(english.intersection(french)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkRecord(self, n):
        BASE = 10 ** 9 + 7
        records = [1, 2]            
        zero, one, two = 1, 1, 0    
        for _ in range(2, n + 1):
            zero, one, two = (zero + one + two) % BASE, zero, one
            records.append((zero + one + two) % BASE)   
        result = records[-1]
        for i in range(n):
            result += records[i] * records[n - 1 - i]   
            result %= BASE
        return result
EOF
def rotLeft(nums, k):
    k = k % len(nums)
    count = start = len(nums)-1
    while count >=0 :
        cur = start
        prev = nums[start]
        while 1:
            nextt = (cur - k) % len(nums)
            nums[nextt], prev = prev, nums[nextt]
            cur  = nextt
            count -= 1
            if start == cur:
                break
        start -= 1
    return a
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nd = input().split()
    n = int(nd[0])
    d = int(nd[1])
    a = list(map(int, input().rstrip().split()))
    result = rotLeft(a, d)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ZigzagIterator(object):
    def __init__(self, v1, v2):
        self.vectors = [v for v in (v1, v2) if v]                   
        self.q = deque((i, 0) for i in range(len(self.vectors)))    
    def next(self):
        vector, index = self.q.popleft()
        if index < len(self.vectors[vector])-1:
            self.q.append((vector, index+1))        
        return self.vectors[vector][index]
    def hasNext(self):
        return bool(self.q)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countPrimes(self, n):
        sieve = [False, False] + [True for _ in range(n-2)]     
        for i in range(2, int(n**0.5) + 1):     
            if sieve[i]:                        
                sieve[i*i:n:i] = [False] * len(sieve[i*i:n:i])  
        return sum(sieve)
EOF
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
    def __repr__(self):
        if self:
            return "{} -> {}".format(self.val, repr(self.next))
class Solution(object):
    def reorderList(self, head):
        if head == None or head.next == None:
            return head
        fast, slow, prev = head, head, None
        while fast != None and fast.next != None:
            fast, slow, prev = fast.next.next, slow.next, slow
        current, prev.next, prev = slow, None, None
        while current != None:
            current.next, prev, current = prev, current, current.next
        l1, l2 = head, prev
        dummy = ListNode(0)
        current = dummy
        while l1 != None and l2 != None:
            current.next, current, l1 = l1, l1, l1.next
            current.next, current, l2 = l2, l2, l2.next
        return dummy.next
EOF
def is_palindrome(s):
    return s == s[::-1]
def palindromeIndex(s):
    ret = -1
    lens = len(s)
    ind = 0
    if is_palindrome(s):
        return ret
    while ind < lens//2:
        if s[ind] != s[lens-ind-1]:
            if s[ind+1] == s[lens-ind-1] and s[ind+2] == s[lens-ind-2]:
                ret = ind
                break
            else:
                ret = lens-ind-1
                break
        ind += 1
    return ret
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = palindromeIndex(s)
    print(result)
EOF
def check_a(number, a):
    for el in a:
        if number % el != 0:
            return False
    return True
def check_b(number, b):
    for el in b:
        if el % number != 0:
            return False
    return True
def getTotalX(a, b):
    res = 0
    max_a = max(a)
    min_b = min(b)
    probe = max_a
    while probe <= min_b:
        if check_a(probe, a) and check_b(probe, b):
            res += 1
        probe += max_a
    return res
if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    a = list(map(int, input().strip().split(' ')))
    b = list(map(int, input().strip().split(' ')))
    total = getTotalX(a, b)
    print(total)
EOF
def climbingLeaderboard(scores, alice):
    output = []
    uniq = list(sorted(list(set(scores)), reverse=True))
    last = len(uniq)
    for cur in alice:
        while last > 0 and cur > uniq[last-1]:
            last -= 1
        if cur == uniq[last-1]:
            output.append(last)
        else:
            output.append(last+1)
    return output
if __name__ == "__main__":
    n = int(input().strip())
    scores = list(map(int, input().strip().split(' ')))
    m = int(input().strip())
    alice = list(map(int, input().strip().split(' ')))
    result = climbingLeaderboard(scores, alice)
    print ("\n".join(map(str, result)))
EOF
def theLoveLetterMystery(s):
    string = list(s)
    res = 0
    first = []
    second = []
    if len(string) % 2 == 1:
        first = list(map(lambda x: ord(x), string[:len(string)//2]))
        first = first[::-1]
        second = list(map(lambda x: ord(x), string[len(string)//2 + 1:]))
    else:
        first = list(map(lambda x: ord(x), string[:len(string)//2 - 1]))
        first = first[::-1]
        second = list(map(lambda x: ord(x), string[len(string)//2 + 1:]))
        res = abs(ord(string[len(string)//2 - 1]) - ord(string[len(string)//2]))
    for ind in range(len(first)):
        if first[ind] != second[ind]:
            res += abs(first[ind] - second[ind])
            first[ind] = min(first[ind], second[ind])
            second[ind] = first[ind]
    return res
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = theLoveLetterMystery(s)
    print(result)
EOF
class Solution(object):
    def canWin(self, s):
        g, g_final = [0], 0
        for p in itertools.imap(len, re.split('-+', s)):
            while len(g) <= p:
                g += min(set(xrange(p)) - {x^y for x, y in itertools.izip(g[:len(g)/2], g[-2:-len(g)/2-2:-1])}),
            g_final ^= g[p]
        return g_final > 0  
class Solution2(object):
    def canWin(self, s):
        lookup = {}
        def canWinHelper(consecutives):                                         
            consecutives = tuple(sorted(c for c in consecutives if c >= 2))     
            if consecutives not in lookup:
                lookup[consecutives] = any(not canWinHelper(consecutives[:i] + (j, c-2-j) + consecutives[i+1:])  
                                           for i, c in enumerate(consecutives)  
                                           for j in xrange(c - 1))              
            return lookup[consecutives]                                         
        return canWinHelper(map(len, re.findall(r'\+\++', s)))
class Solution3(object):
    def canWin(self, s):
        i, n = 0, len(s) - 1
        is_win = False
        while not is_win and i < n:                                     
            if s[i] == '+':
                while not is_win and i < n and s[i+1] == '+':           
                    is_win = not self.canWin(s[:i] + '--' + s[i+2:])    
                    i += 1
            i += 1
        return is_win
EOF
class Solution(object):
    def fairCandySwap(self, A, B):
        diff = (sum(A)-sum(B))//2
        setA = set(A)
        for b in set(B):
            if diff+b in setA:
                return [diff+b, b]
        return []
EOF
def alternatingCharacters(s):
    string = list(s)
    last = string.pop()
    res = 0
    while string:
        newone = string.pop()
        if newone == last:
            res += 1
        else:
            last = newone
    return res
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = alternatingCharacters(s)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wordPattern(self, pattern, str):
        str = str.split()
        if len(str) != len(pattern):        
            return False
        p_to_s, s_to_p = {}, {}
        for w, c in zip(str, pattern):
            if c in p_to_s and p_to_s[c] != w:
                return False
            p_to_s[c] = w
            if w in s_to_p and s_to_p[w] != c:
                return False
            s_to_p[w] = c
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canPartitionKSubsets(self, nums, k):
        total = sum(nums)
        if total % k != 0:
            return False        
        target = total // k
        used = [False] * len(nums)
        nums.sort(reverse = True)
        if nums[0] > target:    
            return False
        def dfs(subsets, last, partial):
            if subsets == 1:    
                return True
            if partial == target:   
                return dfs(subsets - 1, 0, 0)
            for i in range(last, len(nums)):    
                if not used[i] and partial + nums[i] <= target:
                    used[i] = True
                    if dfs(subsets, i + 1, partial + nums[i]):  
                        return True
                    used[i] = False
            return False
        return dfs(k, 0, 0)
class Solution2(object):
    def canPartitionKSubsets(self, nums, k):
        total = sum(nums)
        nums.sort(reverse = True)
        target = total // k
        if total % k != 0 or nums[0] > target:
            return False
        partition = [0 for _ in range(k)]
        def helper(i):                          
            if i == len(nums):
                return True
            for j in range(len(partition)):
                if partition[j] + nums[i] <= target:
                    partition[j] += nums[i]
                    if helper(i + 1):
                        return True
                    partition[j] -= nums[i]
                if partition[j] == 0:           
                    break
            return False
        return helper(0)
EOF
def birthdayCakeCandles(ar):https://www.hackerrank.com/challenges/birthday-cake-candles/problem?h_r=next-challenge&h_v=zen
    highest_candle = count = 0
    for c in ar:
        if c > highest_candle:
            highest_candle = c
            count = 1
        elif c == highest_candle:
            count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(ar)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
def make_change_start(coins, n):
    return make_change(coins, n, 0, {})
def make_change(coins, n, index, memo):
    if n == 0:
        return 1
    if index >= len(coins):
        return 0
    key = str(n) + '-' + str(index)
    if key in memo:
        return memo.get(key)
    res = 0
    amount_with_coin = 0
    while amount_with_coin <= n:
        res += make_change(coins, n - amount_with_coin, index + 1, memo)
        amount_with_coin += coins[index]
    memo.update({key:res})
    return res
n,m = input().strip().split(' ')
n,m = [int(n),int(m)]
coins = [int(coins_temp) for coins_temp in input().strip().split(' ')]
print(make_change_start(coins, n))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isBipartite(self, graph):
        n = len(graph)
        colours = [None] * n            
        for i in range(len(graph)):
            if colours[i] is not None:
                continue
            colours[i] = True
            queue = [i]                 
            while queue:
                v = queue.pop()
                for nbor in graph[v]:
                    if colours[nbor] is None:
                        colours[nbor] = not colours[v]  
                        queue.append(nbor)
                    elif colours[nbor] == colours[v]:   
                        return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minDistance(self, word1, word2):
        def LCS(s, t):
            prev_dp = [0 for _ in range(len(word2) + 1)]
            for i in range(1, len(word1) + 1):  
                dp = [0]
                for j in range(1, len(word2) + 1):  
                    if word1[i - 1] == word2[j - 1]:
                        dp.append(1 + prev_dp[j - 1])
                    else:
                        dp.append(max(dp[-1], prev_dp[j]))
                prev_dp = dp
            return prev_dp[-1]
        return len(word1) + len(word2) - 2 * LCS(word1, word2)
EOF
if __name__ == "__main__":
    n, m = map(int, input().strip().split(' '))
    library = defaultdict(list)
    for ind in range(1, n + 1):
        word = input().strip()
        library[word].append(ind)
    for ind in range(m):
        word = input().strip()
        if len(library[word]) > 0:
            print(" ".join(map(str, library[word])))
        else:
            print("-1")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMin(self, nums):
        left, right = 0, len(nums)-1
        while left < right:
            if nums[left] < nums[right]:    
                break
            mid = (left + right) // 2
            if nums[right] < nums[mid]:
                left = mid + 1      
            elif nums[right] > nums[mid] or nums[left] > nums[mid]:
                right = mid         
            else:                   
                left += 1
                right -= 1
        return nums[left]
EOF
def height(root):
    if root is not None:
        return max(1 + height(root.left), 1 + height(root.right))
    else:
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def transpose(self, A):
        return zip(*A)
EOF
class Solution(object):
    def nextGreatestLetter(self, letters, target):
        i = bisect.bisect_right(letters, target)
        return letters[0] if i == len(letters) else letters[i]
EOF
class SnapshotArray(object):
    def __init__(self, length):
        self.__A = collections.defaultdict(lambda: [(-1, 0)])
        self.__snap_id = 0
    def set(self, index, val):
        self.__A[index].append((self.__snap_id, val))
    def snap(self):
        self.__snap_id += 1
        return self.__snap_id - 1
    def get(self, index, snap_id):
        i = bisect.bisect_right(self.__A[index], (snap_id+1, 0)) - 1
        return self.__A[index][i][1]   
EOF
def towerBreakers(n, m):
    if m == 1:
        return 2
    else:
        return 1 if n%2 == 1 else 2
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n, m = input().strip().split(' ')
        n, m = [int(n), int(m)]
        result = towerBreakers(n, m)
        print(result)
EOF
def triplets(a, b, c):
    a = sorted(list(set(a)))
    b = sorted(list(set(b)))
    c = sorted(list(set(c)))
    res = 0
    for el in b:
        left = bisect_right(a, el)
        right = bisect_right(c, el)
        print("left = {} el = {} right = {}".format(left, el, right))
        res += left * right
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    lenaLenbLenc = input().split()
    lena = int(lenaLenbLenc[0])
    lenb = int(lenaLenbLenc[1])
    lenc = int(lenaLenbLenc[2])
    arra = list(map(int, input().rstrip().split()))
    arrb = list(map(int, input().rstrip().split()))
    arrc = list(map(int, input().rstrip().split()))
    ans = triplets(arra, arrb, arrc)
    fptr.write(str(ans) + '\n')
    fptr.close()
EOF
def occurrences(string, sub):
    res = []
    ind = 0
    while ind < len(string) - len(sub) + 1:
        found = string.find(sub, ind)
        if found != -1:
            res.append(found)
            ind = found + 1
        else:
            break
    return res
def gridSearch(G, P):
    for ind_g in range(len(G) - len(P) + 1):
        cur = -1
        all_occurrences = []
        for ind_p, s_pat in enumerate(P):
            all_occurrences.append(occurrences(G[ind_g + ind_p], s_pat))
        ourset = set(all_occurrences[0])
        for lst in all_occurrences:
            ourset &= set(lst)
        if len(ourset) >= 1:
            return 'YES'
    return 'NO'
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        R, C = input().strip().split(' ')
        R, C = [int(R), int(C)]
        G = []
        G_i = 0
        for G_i in range(R):
            G_t = str(input().strip())
            G.append(G_t)
        r, c = input().strip().split(' ')
        r, c = [int(r), int(c)]
        P = []
        P_i = 0
        for P_i in range(r):
            P_t = str(input().strip())
            P.append(P_t)
        result = gridSearch(G, P)
        print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class StockSpanner(object):
    def __init__(self):
        self.stack = []
    def next(self, price):
        result = 1                          
        while self.stack and price >= self.stack[-1][0]:
            _, count = self.stack.pop()
            result += count
        self.stack.append([price, result])
        return result
EOF
class Solution(object):
    def reformat(self, s):
        def char_gen(start, end, count):
            for c in xrange(ord(start), ord(end)+1):
                c = chr(c)
                for i in xrange(count[c]):
                    yield c
            yield ''
        count = collections.defaultdict(int)
        alpha_cnt = 0
        for c in s:
            count[c] += 1
            if c.isalpha():
                alpha_cnt += 1
        if abs(len(s)-2*alpha_cnt) > 1:
            return ""
        result = []
        it1, it2 = char_gen('a', 'z', count), char_gen('0', '9', count)
        if alpha_cnt < len(s)-alpha_cnt:
            it1, it2 = it2, it1
        while len(result) < len(s):
            result.append(next(it1))
            result.append(next(it2))
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findCircleNum(self, M):
        n = len(M)
        group = [i for i in range(n)]       
        def get_group(x):                   
            while group[x] != x:
                group[x] = group[group[x]]  
                x = group[x]
            return x
        for i in range(1, n):
            for j in range(i):              
                if M[i][j] == 1:
                    group[get_group(i)] = get_group(j)  
        return len(set(get_group(i) for i in range(n)))
class Solution2(object):
    def findCircleNum(self, M):
        def dfs(i):
            for j in range(len(M)):
                if M[i][j] == 1:
                    if j not in seen:
                        seen.add(j)
                        dfs(j)
        circles = 0
        seen = set()
        for i in range(len(M)):
            if i not in seen:
                circles += 1
                dfs(i)
        return circles
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def numUniqueEmails(self, emails):
        unique = set()
        for email in emails:
            local, domain = email.split("
            plus = local.index("+")
            if plus != -1:
                local = local[:plus]
            local = local.replace(".", "")
            unique.add(local + domain)
        return len(unique)
EOF
def introTutorial(V, arr):
    return arr.index(V)
if __name__ == "__main__":
    V = int(input().strip())
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = introTutorial(V, arr)
    print(result)
EOF
def validate(s, first):
    while s:
        if s.startswith(first):
            s = s[len(first):]
            first = str(int(first) + 1)
        else:
            return False
    return True
def separateNumbers(s):
    if s[0] == '0' and len(s) > 1:
        return "NO"
    else:
        for ind in range(1, len(s)//2 + 1):
            first = s[:ind]
            if validate(str(s), first):
                return "YES " + first
    return "NO"
if __name__ == "__main__":
    q = int(input().strip())
    for a0 in range(q):
        s = input().strip()
        print(separateNumbers(s))
EOF
class Solution(object):
    def maxVacationDays(self, flights, days):
        if not days or not flights:
            return 0
        dp = [[0] * len(days) for _ in xrange(2)]
        for week in reversed(xrange(len(days[0]))):
            for cur_city in xrange(len(days)):
                dp[week % 2][cur_city] = days[cur_city][week] + dp[(week+1) % 2][cur_city]
                for dest_city in xrange(len(days)):
                    if flights[cur_city][dest_city] == 1:
                        dp[week % 2][cur_city] = max(dp[week % 2][cur_city], \
                                                     days[dest_city][week] + dp[(week+1) % 2][dest_city])
        return dp[0][0]
EOF
def sequence(N):
    x = N % 8
    if x == 0 or x == 1:
        return N
    if x == 2 or x == 3:
        return 2
    if x == 4 or x == 5:
        return N+2
    if x == 6 or x == 7:
        return 0
def solution(L, R):
    res = 0
    if (R - L) % 2 == 1:
        for ind in range(L+1, R+1, 2):
            res ^= ind
    else:
        for ind in range(L+1):
            res ^= ind
        for ind in range(L+2, R+1, 2):
            res ^= ind
    return res
Q = int(input().strip())
for a0 in range(Q):
    L,R = input().strip().split(' ')
    L,R = [int(L),int(R)]
    print(sequence(L-1)^sequence(R))
EOF
class Solution(object):
    def longestLine(self, M):
        if not M: return 0
        result = 0
        dp = [[[0] * 4 for _ in xrange(len(M[0]))] for _ in xrange(2)]
        for i in xrange(len(M)):
            for j in xrange(len(M[0])):
                dp[i % 2][j][:] = [0] * 4
                if M[i][j] == 1:
                    dp[i % 2][j][0] = dp[i % 2][j - 1][0]+1 if j > 0 else 1
                    dp[i % 2][j][1] = dp[(i-1) % 2][j][1]+1 if i > 0 else 1
                    dp[i % 2][j][2] = dp[(i-1) % 2][j-1][2]+1 if (i > 0 and j > 0) else 1
                    dp[i % 2][j][3] = dp[(i-1) % 2][j+1][3]+1 if (i > 0 and j < len(M[0])-1) else 1
                    result = max(result, max(dp[i % 2][j]))
        return result
EOF
class Solution(object):
    def countBinarySubstrings(self, s):
        result, prev, curr = 0, 0, 1
        for i in xrange(1, len(s)):
            if s[i-1] != s[i]:
                result += min(prev, curr)
                prev, curr = curr, 1
            else:
                curr += 1
        result += min(prev, curr)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MaxStack(object):
    def __init__(self):
        self.stack = [(float("-inf"), float("-inf"))]  
    def push(self, x):
        self.stack.append((x, max(x, self.stack[-1][1])))
    def pop(self):
        x, _ = self.stack.pop()
        return x
    def top(self):
        return self.stack[-1][0]
    def peekMax(self):
        return self.stack[-1][1]
    def popMax(self):
        temp = []
        x, target = self.stack.pop()
        while x != target:
            temp.append(x)
            x, _ = self.stack.pop()
        for x in reversed(temp):
            self.push(x)
        return target
EOF
def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)
n = int(input())
print(fibonacci(n))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reverseWords(self, s):
        return " ".join([w[::-1] for w in s.split(" ")])
EOF
if __name__ == '__main__':
    s = input()
    print(any(char.isalnum() for char in s))
    print(any(char.isalpha() for char in s))
    print(any(char.isdigit() for char in s))
    print(any(char.islower() for char in s))
    print(any(char.isupper() for char in s))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numWays(self, n, k):
        if n == 0:
            return 0
        if n == 1:
            return k
        same, different = 0, k
        for _ in range(n - 1):
            same, different = different, (same + different) * (k - 1)
        return same + different
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countDigitOne(self, n):
        if n <= 0:
            return 0
        ones = 0
        block_size = 10
        for _ in range(len(str(n))):
            blocks, rem = divmod(n + 1, block_size)
            ones += blocks * block_size // 10       
            ones += min(block_size // 10, max(0, rem - block_size // 10))   
            block_size *= 10
        return ones
print(Solution().countDigitOne(524))
EOF
def designerPdfViewer(h, word):
    return len(word) * max(list(map(lambda x: h[ord(x) - ord('a')], word)))
if __name__ == "__main__":
    h = list(map(int, input().strip().split(' ')))
    word = input().strip()
    result = designerPdfViewer(h, word)
    print(result)
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def has_cycle(head):
    fast = slow = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        index = int(input())
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        extra = SinglyLinkedListNode(-1);
        temp = llist.head;
        for i in range(llist_count):
            if i == index:
                extra = temp
            if i != llist_count-1:
                temp = temp.next
        temp.next = extra
        result = has_cycle(llist.head)
        fptr.write(str(int(result)) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        flowerbed.append(0)     
        i = 0
        while n > 0 and i < len(flowerbed) - 1:
            if flowerbed[i + 1] == 1:
                i += 3
            elif flowerbed[i] == 1:
                i += 2
            elif i != 0 and flowerbed[i - 1] == 1:
                i += 1
            else:
                n -= 1
                i += 2
        return n == 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def matrixScore(self, A):
        rows, cols = len(A), len(A[0])
        for r in range(rows):
            if A[r][0] == 0:                
                for c in range(1, cols):    
                    A[r][c] = 1 - A[r][c]
        score = rows * 2 ** (cols - 1)      
        for c in range(1, cols):            
            col_count = sum(A[r][c] for r in range(rows))       
            best_col_count = max(col_count, rows - col_count)   
            col_val = 2 ** ((cols - 1) - c)                     
            score += col_val * best_col_count
        return score
EOF
def _digitSum(number):
    if len(number) == 1:
        return int(number)
    else:
        temp = str(sum([int(x) for x in number]))
        return _digitSum(temp)
def digitSum(number, k):
    temp = str(k*sum([int(x) for x in number]))
    return _digitSum(temp)
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [str(n), int(k)]
    result = digitSum(n, k)
    print(result)
EOF
class Solution(object):
    def getPermutation(self, n, k):
        seq, k, fact = "", k - 1, math.factorial(n - 1)
        perm = [i for i in xrange(1, n + 1)]
        for i in reversed(xrange(n)):
            curr = perm[k / fact]
            seq += str(curr)
            perm.remove(curr)
            if i > 0:
                k %= fact
                fact /= i
        return seq
EOF
class Solution(object):
    def numSmallerByFrequency(self, queries, words):
        words_freq = sorted(word.count(min(word)) for word in words)
        return [len(words)-bisect.bisect_right(words_freq, query.count(min(query))) \
                for query in queries]
EOF
def missingNumbers(arr, brr):
    acnt = Counter(arr)
    bcnt = Counter(brr)
    for el in acnt.items():
        get = bcnt.get(el[0])
        if get:
            bcnt[el[0]] -= el[1]
    bcnt = list(map(lambda x: x[0], (filter(lambda x: x[1] > 0, bcnt.items()))))
    bcnt = sorted(bcnt)
    return bcnt
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    m = int(input().strip())
    brr = list(map(int, input().strip().split(' ')))
    result = missingNumbers(arr, brr)
    print (" ".join(map(str, result)))
EOF
def flatlandSpaceStations(n, stations):
    stations = sorted(stations)
    res = stations[0]
    for ind in range(1, len(stations)):
        res = max(res, (stations[ind] - stations[ind-1])//2)
    res = max(res, n-1 - stations[-1])
    return res
if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    c = list(map(int, input().strip().split(' ')))
    result = flatlandSpaceStations(n, c)
    print(result)
EOF
def misereNim(pile):
    if set(pile) == {1}:
        if len(pile)%2 == 0:
            return 'First'
        else:
            return 'Second'
    res = reduce((lambda x, y: x ^ y), pile)
    if res == 0:
        return 'Second'
    else:
        return 'First'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        s = list(map(int, input().rstrip().split()))
        result = misereNim(s)
        fptr.write(result + '\n')
    fptr.close()
EOF
def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
def almostSorted(arr):
    swap_l = -1
    swap_r = -1
    for ind in range(1, len(arr)):
        if arr[ind - 1] > arr[ind]:
            swap_l = ind - 1
            break
    for ind in range(swap_l + 1, len(arr)):
        if ind == len(arr) - 1 or arr[ind + 1] > arr[swap_l]:
            swap_r = ind
            arr[swap_l], arr[swap_r] = arr[swap_r], arr[swap_l]
            break
    if is_sorted(arr):
        print("yes")
        print("swap {} {}".format(swap_l + 1, swap_r + 1))
        return True
    arr[swap_l], arr[swap_r] = arr[swap_r], arr[swap_l]
    rev_l = -1
    rev_r = -1
    for ind in range(len(arr) - 1):
        if rev_l == -1 and arr[ind] > arr[ind + 1]:
            rev_l = ind
        elif rev_l != -1 and arr[ind] < arr[ind + 1]:
            rev_r = ind
            break
    to_rev = arr[rev_l:rev_r+1]
    arr = arr[:rev_l] + to_rev[::-1] + arr[rev_r+1:]
    if is_sorted(arr):
        print("yes")
        print("reverse {} {}".format(rev_l + 1, rev_r + 1))
        return True
    print("no")
    return False
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    almostSorted(arr)
EOF
class Solution(object):
    def maxNumberOfBalloons(self, text):
        TARGET = "balloon"
        source_count = collections.Counter(text)
        target_count = collections.Counter(TARGET)
        return min(source_count[c]//target_count[c] for c in target_count.iterkeys())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findSecondMinimumValue(self, root):
        min_val = root.val  
        self.second_min = float("inf")
        def helper(node):
            if not node:
                return
            if node.val == min_val:
                helper(node.left)
                helper(node.right)
            else:  
                self.second_min = min(node.val, self.second_min)
        helper(root)
        return -1 if self.second_min == float("inf") else self.second_min
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProfit(self, prices, fee):
        buy, sell = float("-inf"), 0
        for price in prices:
            buy, sell = max(buy, sell - price), max(sell, buy + price - fee)
        return sell
EOF
class DoublyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
        self.prev = None
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = DoublyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
            node.prev = self.tail
        self.tail = node
def print_doubly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def sortedInsert(head, data):
    node = DoublyLinkedListNode(data)
    if data <= head.data:
        node.next = head
        head.prev = node
        head = node
        return head
    cur = head
    while cur.next:
        if cur.next.data > data:
            node.next = cur.next
            cur.next.prev = node
            node.prev = cur
            cur.next = node
            return head
        cur = cur.next
    cur.next = node
    node.prev = cur
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        llist_count = int(input())
        llist = DoublyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        data = int(input())
        llist1 = sortedInsert(llist.head, data)
        print_doubly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxKilledEnemies(self, grid):
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        max_kill_enemies, row_kill, col_kill = 0, 0, [0 for _ in range(cols)]
        for r in range(rows):
            for c in range(cols):
                if c == 0 or grid[r][c - 1] == "W":             
                    row_kill, i = 0, c
                    while i < cols and grid[r][i] != "W":       
                        row_kill +=  grid[r][i] == "E"          
                        i += 1
                if r == 0 or grid[r - 1][c] == "W":             
                    col_kill[c], i = 0, r
                    while i < rows and grid[i][c] != "W":       
                        col_kill[c] +=  grid[i][c] == "E"       
                        i += 1
                if grid[r][c] == "0":
                    max_kill_enemies = max(max_kill_enemies, row_kill + col_kill[c])
        return max_kill_enemies
EOF
if __name__ == "__main__":
    k = int(input().strip())
    numbers = list(map(int, input().strip().split(' ')))
    print((sum(set(numbers))*k - sum(numbers))//(k-1))
EOF
class Solution(object):
    def maxScore(self, s):
        result, zeros, ones = 0, 0, 0
        for i in xrange(1, len(s)-1):
            if s[i] == '0':
                zeros += 1
            else:
                ones += 1
            result = max(result, zeros-ones)
        return result + ones + (s[0] == '0') + (s[-1] == '1')
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        prev = result = ListNode(None)      
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            prev.next = ListNode(carry % 10)
            prev = prev.next
            carry //= 10
        return result.next
EOF
n = int(input().strip())
heights = sorted([int(i) for i in input().split()][:n], reverse = True)
count = 1
for i in range(1, len(heights)):
    if heights[i] == heights[0]:
        count += 1
print(count)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numsSameConsecDiff(self, N, K):
        partials = [i for i in range(1, 10)]
        for _ in range(N - 1):
            new_partials = []
            for p in partials:
                last_digit = p % 10
                if last_digit - K >= 0:
                    new_partials.append(p * 10 + last_digit - K)
                if K != 0 and last_digit + K < 10:
                    new_partials.append(p * 10 + last_digit + K)
            partials = new_partials
        if N == 1:
            partials.append(0)
        return partials
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def strStr(self, haystack, needle):
        for i in range(len(haystack)-len(needle)+1):
            for j in range(len(needle)):
                if haystack[i+j] != needle[j]:
                    break
            else:           
                return i
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def triangleNumber(self, nums):
        nums.sort()
        triangles = 0
        for i, longest_side in enumerate(nums):
            left, right = 0, i - 1
            while left < right:
                shortest_side, middle_side = nums[left], nums[right]
                if shortest_side + middle_side > longest_side:  
                    triangles += right - left  
                    right -= 1  
                else:
                    left += 1  
        return triangles
class Solution2(object):
    def triangleNumber(self, nums):
        sides = Counter(nums)
        if 0 in sides:
            del sides[0]
        sides = list(sides.items())  
        sides.sort()
        triangles = 0
        def binom(n, k):
            if k > n:
                return 0
            return factorial(n) // (factorial(n - k) * factorial(k))
        for i, (s1, c1) in enumerate(sides):
            for j, (s2, c2) in enumerate(sides[i:]):
                j2 = j + i
                for s3, c3 in sides[j2:]:
                    if s1 == s2 == s3:  
                        triangles += binom(c1, 3)
                    elif s1 == s2:      
                        if s1 + s2 > s3:
                            triangles += c3 * binom(c1, 2)
                    elif s2 == s3:      
                        triangles += c1 * binom(c2, 2)
                    else:   
                        if s1 + s2 > s3:
                            triangles += c1 * c2 * c3
        return triangles
EOF
def camelcase(s):
    count = 0
    for i in s:
        if i.isupper():
            count += 1
    return count + 1
if __name__ == "__main__":
    s = input().strip()
    result = camelcase(s)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxAreaOfIsland(self, grid):
        rows, cols = len(grid), len(grid[0])
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        max_area = 0
        def island_area(r, c):
            grid[r][c] = 0
            area = 1
            for dr, dc in neighbours:       
                if 0 <= r + dr < rows and 0 <= c + dc < cols and grid[r + dr][c + dc] == 1:
                    area += island_area(r + dr, c + dc)
            return area
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1:     
                    max_area = max(max_area, island_area(row, col))
        return max_area
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestConsecutive(self, nums):
        numset = set(nums)          
        longest = 0
        for num in numset:
            if num-1 in numset:     
                continue
            seq = 0
            while num in numset:    
                seq += 1
                num += 1
            longest = max(longest, seq)
        return longest
EOF
d = deque()
for i in range(int(input())):
    q = list(map(str,input().split()))
    if q[0] == "append":
        d.append(int(q[1]))
    elif q[0] == "pop":
        d.pop()
    elif q[0] == "popleft":
        d.popleft()
    elif q[0] == "appendleft":
        d.appendleft(int(q[1]))
print(*d)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def atMostNGivenDigitSet(self, D, N):
        S = str(N)
        K = len(S)
        dp = [0] * K + [1]  
        for i in range(K - 1, -1, -1):
            for d in D:
                if d < S[i]:    
                    dp[i] += len(D) ** (K - i - 1)
                elif d == S[i]: 
        return dp[0] + sum(len(D) ** i for i in range(1, K))    
EOF
class Node:
    def __init__(self, info): 
        self.info = info  
        self.left = None  
        self.right = None 
        self.level = None 
    def __str__(self):
        return str(self.info) 
class BinarySearchTree:
    def __init__(self): 
        self.root = None
    def create(self, val):  
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break
def inOrder(root):
    if not root:
        return None
    inOrder(root.left)
    print(root.info,end=" ")
    inOrder(root.right)
tree = BinarySearchTree()
t = int(input())
arr = list(map(int, input().split()))
for i in range(t):
    tree.create(arr[i])
inOrder(tree.root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reversePairs(self, nums):
        self.pairs = 0
        def mergesort(nums):
            if len(nums) < 2:
                return nums
            mid = len(nums) // 2
            left = mergesort(nums[:mid])
            right = mergesort(nums[mid:])
            return merge(left, right)
        def merge(left, right):
            j = 0
            for num in left:
                while j < len(right) and num > 2 * right[j]:
                    j += 1
                self.pairs += j     
            merged = []
            i, j = 0, 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
            return merged + left[i:] + right[j:]
        mergesort(nums)
        return self.pairs
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minFallingPathSum(self, A):
        n = len(A)
        row_minima = [0] * n
        for r in range(n - 1, -1, -1):      
            new_row_minima = list(A[r])     
            for c in range(n):
                new_row_minima[c] += min(row_minima[max(0, c - 1):c + 2])   
            row_minima = new_row_minima
        return min(row_minima)              
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def carFleet(self, target, position, speed):
        fleets = 0
        previous = -1                       
        cars = zip(position, speed)
        cars.sort(reverse = True)           
        for pos, spd in cars:
            time = (target - pos) / float(spd)  
            if time > previous:
                fleets += 1
                previous = time                 
        return fleets
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getSkyline(self, buildings):
        skyline = [(0, 0)]              
        current = [(0, float('inf'))]   
        edges = [(l, -h, r) for l, r, h in buildings]       
        edges += [(r, 0, None) for _, r, _ in buildings]    
        edges.sort()
        for x, neg_h, r in edges:
            while current[0][1] <= x:       
                heapq.heappop(current)      
            if neg_h != 0:                  
                heapq.heappush(current, (neg_h, r))
            if skyline[-1][1] != -current[0][0]:        
                skyline.append([x, -current[0][0]])     
        return skyline[1:]
EOF
def factorial(n):
    if n is 1:
        return 1
    else:
        return n * factorial(n-1)
if __name__ == "__main__":
    n = int(input().strip())
    result = factorial(n)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def uniqueLetterString(self, S):
        unique = 0
        indices = [[-1] for _ in range(26)]
        for i, c in enumerate(S):
            indices[ord(c) - ord("A")].append(i)
        for index_list in indices:
            index_list.append(len(S))
            for i in range(1, len(index_list) - 1):
                unique += (index_list[i] - index_list[i - 1]) * (index_list[i + 1] - index_list[i])
        return unique % (10 ** 9 + 7)
EOF
if __name__ == '__main__':
    n = int(input())
    arr = list(set(map(int, input().split())))
    print(sorted(arr)[-2])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def toLowerCase(self, str):
        diff = ord("a") - ord("A")
        return "".join(chr(ord(c) + diff) if "A" <= c <= "Z" else c for c in str)
EOF
returnDate = list(map(int, input().split()))
dueDate = list(map(int, input().split()))
fine = 0
if returnDate[2] > dueDate[2]:
    fine = 10000
elif returnDate[2] == dueDate[2]:
    if returnDate[1] > dueDate[1]:
        fine = 500 * (returnDate[1] - dueDate[1])
    elif returnDate[1] == dueDate[1]:
        if returnDate[0] > dueDate[0]:
            fine = 15 * (returnDate[0] - dueDate[0])
print(fine)
EOF
def is_leap(year):
    return bool(year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
year = int(input())
print(is_leap(year))
EOF
m, english = input(), set(map(int, input().split()))
n, french = input(), set(map(int, input().split()))
print(len(english.difference(french)))
EOF
input_string = input()
print('Hello, World.')
print(input_string)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canMeasureWater(self, x, y, z):
        def gcd(a, b):
            while b != 0:
                a, b = b, a % b
            return a
        if z == 0:
            return True
        g = gcd(x, y)
        if g == 0:
            return False
        return z % g == 0 and z <= x + y
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isRectangleOverlap(self, rec1, rec2):
        x_overlap = min(max(0, rec1[2] - rec2[0]), max(0, rec2[2] - rec1[0]))
        if x_overlap == 0:
            return False
        return min(max(0, rec1[3] - rec2[1]), max(0, rec2[3] - rec1[1])) > 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isAlienSorted(self, words, order):
        indices = {c: i for i, c in enumerate(order)}
        prev = []
        for word in words:
            mapping = [indices[c] for c in word]
            if mapping < prev:
                return False
            prev = mapping
        return True
EOF
if __name__ == "__main__":
    cnum = complex(input().strip())
    print(abs(cnum))
    print(cmath.phase(cnum))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def searchBST(self, root, val):
        if not root:                
            return None
        if root.val == val:
            return root
        if val > root.val:
            return self.searchBST(root.right, val)
        return self.searchBST(root.left, val)
EOF
def rightRotation(a, d):
    out = list(a)
    a_len = len(a)
    for ind, el in enumerate(a):
        out[(ind + d) % a_len] = el
    return out
def circularArrayRotation(a, m):
    out = []
    for pos in m:
        out.append(a[pos])
    return out
if __name__ == "__main__":
    n, k, q = input().strip().split(' ')
    n, k, q = [int(n), int(k), int(q)]
    a = list(map(int, input().strip().split(' ')))
    m = []
    m_i = 0
    for m_i in range(q):
        m_t = int(input().strip())
        m.append(m_t)
    a = rightRotation(a, k)
    result = circularArrayRotation(a, m)
    print ("\n".join(map(str, result)))
EOF
def miniMaxSum(arr):
    m = sum(arr)
    print(m-max(arr),m-min(arr))
if __name__ == '__main__':
    arr = list(map(int, input().rstrip().split()))
    miniMaxSum(arr)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def candy(self, ratings):
        left = [1 for _ in range(len(ratings))]     
        right = [1 for _ in range(len(ratings))]
        for i in range(1, len(ratings)):            
            if ratings[i] > ratings[i-1]:
                left[i] = left[i-1] + 1
        candies = left[-1]      
        for i in range(len(ratings)-2, -1, -1):
            if ratings[i] > ratings[i+1]:           
                right[i] = right[i+1] + 1
            candies += max(left[i], right[i])       
        return candies
EOF
def gameOfStones(n):
    if n%7 == 1 or n%7 == 0:
        return 'Second'
    else:
        return 'First'
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = int(input().strip())
        result = gameOfStones(n)
        print(result)
EOF
class Solution(object):
    def isPerfectSquare(self, num):
        left, right = 1, num
        while left <= right:
            mid = left + (right - left) / 2
            if mid >= num / mid:
                right = mid - 1
            else:
                left = mid + 1
        return left == num / left and num % left == 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestWord(self, words):
        length_to_words = defaultdict(set)
        for word in words:      
            length_to_words[len(word)].add(word)
        candidates = {""}       
        length = 0              
        while True:
            next_candidates = set()
            for longer_word in length_to_words[length + 1]: 
                if longer_word[:-1] in candidates:
                    next_candidates.add(longer_word)
            if not next_candidates:
                return sorted(list(candidates))[0]          
            length += 1
            candidates = next_candidates
EOF
def check_diag(queen, obst):
    if queen[1] == obst[1]:
        return 0
    check = (queen[0] - obst[0])/(queen[1] - obst[1])
    if fabs(check) == 1.0:
        return int(check)
    else:
        return 0
def queensAttack(n, k, r_q, c_q, obstacles):
    queen = [r_q, c_q]
    res = 0
    obst_by_row = list(filter(lambda x: x[0] == r_q, obstacles))
    obst_by_col = list(filter(lambda x: x[1] == c_q, obstacles))
    obst_by_plus_diag = list(filter(lambda x: check_diag(queen, x) == 1, obstacles))
    obst_by_neg_diag = list(filter(lambda x: check_diag(queen, x) == -1, obstacles))
    if not obst_by_col:
        res += n-1
    else:
        obst_higher = list(filter(lambda x: x[0] > r_q, obst_by_col)) 
        if obst_higher:
            min_higher = min(obst_higher, key = lambda x: x[0])[0]
        else:
            min_higher = n+1
        obst_lower = list(filter(lambda x: x[0] < r_q, obst_by_col)) 
        if obst_lower:
            max_lower = max(obst_lower, key = lambda x: x[0])[0]
        else:
            max_lower = 0
        res += min_higher - max_lower - 2
    if not obst_by_row:
        res += n-1
    else:
        obst_higher = list(filter(lambda x: x[1] > c_q, obst_by_row)) 
        if obst_higher:
            min_higher = min(obst_higher, key = lambda x: x[1])[1]
        else:
            min_higher = n+1
        obst_lower = list(filter(lambda x: x[1] < c_q, obst_by_row)) 
        if obst_lower:
            max_lower = max(obst_lower, key = lambda x: x[1])[1]
        else:
            max_lower = 0
        res += min_higher - max_lower - 2
    if not obst_by_plus_diag:
        res += n-1 - abs(r_q - c_q)
    else:
        obst_higher = list(filter(lambda x: x[0] > r_q, obst_by_plus_diag))
        if obst_higher:
            min_higher = min(obst_higher, key = lambda x: x[0])[0]
        else:
            min_higher = n+1
        obst_lower = list(filter(lambda x: x[0] < r_q, obst_by_plus_diag)) 
        if obst_lower:
            max_lower = max(obst_lower, key = lambda x: x[0])[0]
        else:
            max_lower = 0
        res += min_higher - max_lower - 2 - abs(r_q - c_q)
    if not obst_by_neg_diag:
        res += min(n - c_q, r_q - 1)
        res += min(c_q - 1, n - r_q)
    else:
        obst_higher = list(filter(lambda x: x[0] > r_q, obst_by_neg_diag))
        if obst_higher:
            min_higher = min(obst_higher, key = lambda x: x[0])[1]
        else:
            min_higher = n+1
        obst_lower = list(filter(lambda x: x[0] < r_q, obst_by_neg_diag)) 
        if obst_lower:
            max_lower = max(obst_lower, key = lambda x: x[0])[1]
        else:
            max_lower = 0
        print("high = {} low = {}".format(min_higher, max_lower))
        print("r_q = {} c_q = {}".format(r_q, c_q))
        if max_lower != 0:
            res += max_lower - c_q - 1
        if min_higher != n+1:
            res += c_q - min_higher - 1
    return res
def queensAttack_naive(n, k, r_q, c_q, obstacles):
    obst_by_row = list(filter(lambda x: x[0] == r_q, obstacles))
    obst_by_col = list(filter(lambda x: x[1] == c_q, obstacles))
    obs_dict = gen_obs_dict(obstacles)
    res = 0
    if not obst_by_col:
        res += n-1
    else:
        for row_ind in range(r_q+1, n+1):
            key = str(row_ind) + "-" + str(c_q)
            if obs_dict[key] != -1:
                res += 1
            else:
                break
        for row_ind in range(r_q-1, 0, -1):
            key = str(row_ind) + "-" + str(c_q)
            if obs_dict[key] != -1:
                res += 1
            else:
                break
    if not obst_by_row:
        res += n-1
    else:
        for col_ind in range(c_q+1, n+1):
            key = str(r_q) + "-" + str(col_ind)
            if obs_dict[key] != -1:
                res += 1
            else:
                break
        for col_ind in range(c_q-1, 0, -1):
            key = str(r_q) + "-" + str(col_ind)
            if obs_dict[key] != -1:
                res += 1
            else:
                break
    row_ind, col_ind = r_q+1, c_q+1
    while col_ind != 0 and row_ind != 0 and col_ind != n+1 and row_ind != n+1:
        key = str(row_ind) + "-" + str(col_ind)
        if obs_dict[key] != -1:
            res += 1
            row_ind += 1
            col_ind += 1
        else:
            break
    row_ind, col_ind = r_q-1, c_q+1
    while col_ind != 0 and row_ind != 0 and col_ind != n+1 and row_ind != n+1:
        key = str(row_ind) + "-" + str(col_ind)
        if obs_dict[key] != -1:
            res += 1
            row_ind -= 1
            col_ind += 1
        else:
            break
    row_ind, col_ind = r_q+1, c_q-1
    while col_ind != 0 and row_ind != 0 and col_ind != n+1 and row_ind != n+1:
        key = str(row_ind) + "-" + str(col_ind)
        if obs_dict[key] != -1:
            res += 1
            row_ind += 1
            col_ind -= 1
        else:
            break        
    row_ind, col_ind = r_q-1, c_q-1
    while col_ind != 0 and row_ind != 0 and col_ind != n+1 and row_ind != n+1:
        key = str(row_ind) + "-" + str(col_ind)
        if obs_dict[key] != -1:
            res += 1
            row_ind -= 1
            col_ind -= 1
        else:
            break        
    return res
def gen_obs_dict(obstacles):
    dict_out = defaultdict(int)    
    for obs in obstacles:
        row, col = obs[0], obs[1]
        key = str(row) + "-" + str(col)
        dict_out[key] = -1
    return dict_out
if __name__ == "__main__":
    n, k = [int(x) for x in input().strip().split(' ')]
    r_q, c_q = [int(x) for x in input().strip().split(' ')]
    obstacles = []
    for obstacles_i in range(k):
        obstacles_t = [int(obstacles_temp) for obstacles_temp in input().strip().split(' ')]
        obstacles.append(obstacles_t)
    result = queensAttack_naive(n, k, r_q, c_q, obstacles)
    print(result)
EOF
class Solution(object):
    def canConvert(self, str1, str2):
        if str1 == str2:
            return True
        lookup = {}
        for i, j in itertools.izip(str1, str2):
            if lookup.setdefault(i, j) != j:
                return False
        return len(set(str2)) < 26
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ValidWordAbbr(object):
    def __init__(self, dictionary):
        self.dictionary = set(dictionary)
        self.freq = defaultdict(int)
        for word in self.dictionary:
            self.freq[self.abbreviate(word)] += 1
    def isUnique(self, word):
        abbr = self.abbreviate(word)
        if word in self.dictionary:
            return self.freq[abbr] == 1
        else:
            return abbr not in self.freq
    def abbreviate(self, word):
        n = len(word)
        if n < 3:
            return word
        return word[0] + str(n - 2) + word[-1]
EOF
class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def comparator(a, b):
        if (a.score < b.score):
            return 1
        if (a.score > b.score):
            return -1
        if (a.name < b.name):
            return -1
        if (a.name > b.name):
            return 1
        return 0
EOF
def minimumAbsoluteDifference(n, arr):
    arr = sorted(arr)
    res = 10**9
    for ind in range(1, len(arr)):
        res = min(res, arr[ind] - arr[ind-1])
    return res
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    result = minimumAbsoluteDifference(n, arr)
    print(result)
EOF
class Solution(object):
    def isPowerOfTwo(self, n):
        return n > 0 and (n & (n - 1)) == 0
class Solution2(object):
    def isPowerOfTwo(self, n):
        return n > 0 and (n & ~-n) == 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def kSimilarity(self, A, B):
        visited = set()
        k = 0
        frontier = {A}
        while True:
            if B in frontier:
                return k
            new_frontier = set()
            for word in frontier:
                if word in visited:
                    continue
                i = 0
                while word[i] == B[i]:              
                    i += 1
                for j in range(i + 1, len(A)):
                    if word[j] != B[i]:             
                        continue
                    swapped = word[:i] + word[j] + word[i + 1:j] + word[i] + word[j + 1:]   
                    new_frontier.add(swapped)
            k += 1
            visited |= frontier                     
            frontier = new_frontier
EOF
def repeatedString(s, n):
    l = len(s)
    count = 0
    for i in s:
        if i == "a":
            count += 1
    if count == 0:
        return 0
    res = count * (n//l)
    for i in range(n%l):
        if s[i] == "a":
            res += 1
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    n = int(input())
    result = repeatedString(s, n)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canTransform(self, start, end):
        if len(start) != len(end):      
            return False
        left, right = 0, 0              
        for c1, c2 in zip(start, end):  
            if c1 == "L":
                left += 1
            elif c1 == "R":
                right += 1
            if c2 == "L":
                left -= 1
            elif c2 == "R":
                right -= 1
            if left > 0 or right < 0:   
                return False
            if left < 0 and right > 0:  
                return False
        return left == 0 and right == 0 
EOF
if __name__ == "__main__":
    a = int(input().strip())
    b = int(input().strip())
    c = int(input().strip())
    d = int(input().strip())
    print(pow(a, b) + pow(c, d))
EOF
class Solution(object):
    def deckRevealedIncreasing(self, deck):
        d = collections.deque()
        deck.sort(reverse=True)
        for i in deck:
            if d:
                d.appendleft(d.pop())
            d.appendleft(i)
        return list(d)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def coinChange(self, coins, amount):
        coins.sort(reverse = True)
        self.result = float("inf")
        def dfs(largest_coin, remainder, used_coins):
            if remainder == 0:
                self.result = min(self.result, used_coins)
            for i in range(largest_coin, len(coins)):                   
                if remainder >= coins[i] * (self.result - used_coins):  
                    break
                if coins[i] <= remainder:                               
                    dfs(i, remainder - coins[i], used_coins + 1)
        dfs(0, amount, 0)
        return self.result if self.result != float("inf") else -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isHappy(self, n):
        if n == 1:
            return True, 1
        slow = sum([int(c)*int(c) for c in str(n)])
        fast = sum([int(c)*int(c) for c in str(slow)])
        while fast != slow:
            slow = sum([int(c)*int(c) for c in str(slow)])
            fast = sum([int(c)*int(c) for c in str(fast)])
            fast = sum([int(c)*int(c) for c in str(fast)])
        return slow == 1
class Solution2(object):
    def isHappy(self, n):
        while True:
            if n == 1:
                return True
            if n == 4:
                return False
            n = sum([int(c)*int(c) for c in str(n)])
EOF
class HtmlParser(object):
   def getUrls(self, url):
       pass
class Solution(object):
    def crawl(self, startUrl, htmlParser):
        SCHEME = "http://"
        def hostname(url):
            pos = url.find('/', len(SCHEME))
            if pos == -1:
                return url
            return url[:pos]
        result = [startUrl]
        lookup = set(result)
        for from_url in result:
            name = hostname(from_url)
            for to_url in htmlParser.getUrls(from_url):
                if to_url not in lookup and name == hostname(to_url):
                    result.append(to_url)
                    lookup.add(to_url)
        return result
EOF
def sansaXor(arr):
    res = 0
    arr_len = len(arr)
    if arr_len % 2 == 0:
        return 0
    for ind in range(0, arr_len, 2):
        res ^= arr[ind]
    return res
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = int(input().strip())
        arr = list(map(int, input().strip().split(' ')))
        result = sansaXor(arr)
        print(result)
EOF
def marcsCakewalk(arr):
    arr = sorted(arr, key = lambda x: -x)
    return sum([el * (2**mult) for mult, el in enumerate(arr)])
if __name__ == "__main__":
    n = int(input().strip())
    calorie = list(map(int, input().strip().split(' ')))
    result = marcsCakewalk(calorie)
    print(result)
EOF
def minimumSwaps(arr):
    res = 0
    ind = 0
    while ind < len(arr)-1:
        if arr[ind] == ind + 1:
            ind += 1
            continue
        else:
            arr[arr[ind]-1], arr[ind] = arr[ind], arr[arr[ind]-1]
            res += 1
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    res = minimumSwaps(arr)
    fptr.write(str(res) + '\n')
    fptr.close()
EOF
class Solution(object):
    def largestSumAfterKNegations(self, A, K):
        def kthElement(nums, k, compare):
            def PartitionAroundPivot(left, right, pivot_idx, nums, compare):
                new_pivot_idx = left
                nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
                for i in xrange(left, right):
                    if compare(nums[i], nums[right]):
                        nums[i], nums[new_pivot_idx] = nums[new_pivot_idx], nums[i]
                        new_pivot_idx += 1
                nums[right], nums[new_pivot_idx] = nums[new_pivot_idx], nums[right]
                return new_pivot_idx
            left, right = 0, len(nums) - 1
            while left <= right:
                pivot_idx = random.randint(left, right)
                new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums, compare)
                if new_pivot_idx == k:
                    return
                elif new_pivot_idx > k:
                    right = new_pivot_idx - 1
                else:  
                    left = new_pivot_idx + 1
        kthElement(A, K, lambda a, b: a < b)
        remain = K
        for i in xrange(K):
            if A[i] < 0:
                A[i] = -A[i]
                remain -= 1
        return sum(A) - ((remain)%2)*min(A)*2
class Solution2(object):
    def largestSumAfterKNegations(self, A, K):
        A.sort()
        remain = K
        for i in xrange(K):
            if A[i] >= 0:
                break
            A[i] = -A[i]
            remain -= 1
        return sum(A) - (remain%2)*min(A)*2
EOF
def stringConstruction(s):
    return len(set(s))
if __name__ == "__main__":
    q = int(input().strip())
    for a0 in range(q):
        s = input().strip()
        result = stringConstruction(s)
        print(result)
EOF
class RecentCounter(object):
    def __init__(self):
        self.__q = collections.deque()
    def ping(self, t):
        self.__q.append(t)
        while self.__q[0] < t-3000:
            self.__q.popleft()
        return len(self.__q)
EOF
def extraLongFactorials(n):
    return factorial(n)
if __name__ == "__main__":
    n = int(input().strip())
    print(extraLongFactorials(n))
EOF
answers = [1]
def gen_answers():
    newlen = 1
    for i in range(61):
        if i % 2 == 1:
            newlen += 1
        else:
            newlen *= 2
        answers.append(newlen)
def utopianTree(n):
    return answers[n]
if __name__ == "__main__":
    t = int(input().strip())
    gen_answers()
    for a0 in range(t):
        n = int(input().strip())
        result = utopianTree(n)
        print(result)
EOF
class AhoNode(object):
    def __init__(self):
        self.children = collections.defaultdict(AhoNode)
        self.indices = []
        self.suffix = None
        self.output = None
class AhoTrie(object):
    def step(self, letter):
        while self.__node and letter not in self.__node.children:
            self.__node = self.__node.suffix
        self.__node = self.__node.children[letter] if self.__node else self.__root
        return self.__get_ac_node_outputs(self.__node)
    def reset(self):
        self.__node = self.__root
    def __init__(self, patterns):
        self.__root = self.__create_ac_trie(patterns)
        self.__node = self.__create_ac_suffix_and_output_links(self.__root)
    def __create_ac_trie(self, patterns):  
        root = AhoNode()
        for i, pattern in enumerate(patterns):
            node = root
            for c in pattern:
                node = node.children[c]
            node.indices.append(i)
        return root
    def __create_ac_suffix_and_output_links(self, root):  
        queue = collections.deque()
        for node in root.children.itervalues():
            queue.append(node)
            node.suffix = root
        while queue:
            node = queue.popleft()
            for c, child in node.children.iteritems():
                queue.append(child)
                suffix = node.suffix
                while suffix and c not in suffix.children:
                    suffix = suffix.suffix
                child.suffix = suffix.children[c] if suffix else root
                child.output = child.suffix if child.suffix.indices else child.suffix.output
        return root
    def __get_ac_node_outputs(self, node):  
        result = []
        for i in node.indices:
            result.append(i)
        output = node.output
        while output:
            for i in output.indices:
                result.append(i)
            output = output.output
        return result
class Solution(object):
    def stringMatching(self, words):
        trie = AhoTrie(words)
        lookup = set()
        for i in xrange(len(words)):
            trie.reset()
            for c in words[i]:
                for j in trie.step(c):
                    if j != i:
                        lookup.add(j)
        return [words[i] for i in lookup]
class Solution2(object):
    def stringMatching(self, words):
        def getPrefix(pattern):
            prefix = [-1]*len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j != -1 and pattern[j+1] != pattern[i]:
                    j = prefix[j]
                if pattern[j+1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix
        def kmp(text, pattern, prefix):
            if not pattern:
                return 0
            if len(text) < len(pattern):
                return -1
            j = -1
            for i in xrange(len(text)):
                while j != -1 and pattern[j+1] != text[i]:
                    j = prefix[j]
                if pattern[j+1] == text[i]:
                    j += 1
                if j+1 == len(pattern):
                    return i-j
            return -1
        result = []
        for i, pattern in enumerate(words):
            prefix = getPrefix(pattern)
            for j, text in enumerate(words):
                if i != j and kmp(text, pattern, prefix) != -1:
                    result.append(pattern)
                    break
        return result
class Solution3(object):
    def stringMatching(self, words):
        result = []
        for i, pattern in enumerate(words):
            for j, text in enumerate(words):
                if i != j and pattern in text:
                    result.append(pattern)
                    break
        return result
EOF
def plusMinus(arr):
    arrlen = len(arr)
    positive = negative = zeros = 0
    for elem in arr:
        if elem > 0:
            positive += 1
        elif elem < 0:
            negative += 1
        else:
            zeros += 1
    print("{:.6f}\n{:.6f}\n{:.6f}".format(positive/arrlen, negative/arrlen, zeros/arrlen))
if __name__ == "__main__":
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    plusMinus(arr)
EOF
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Codec(object):
    def encode(self, root):
        def encodeHelper(root, parent, index):
            if not root:
                return None
            node = TreeNode(root.val)
            if index+1 < len(parent.children):
                node.left = encodeHelper(parent.children[index+1], parent, index+1)
            if root.children:
                node.right = encodeHelper(root.children[0], root, 0)
            return node
        if not root:
            return None
        node = TreeNode(root.val)
        if root.children:
            node.right = encodeHelper(root.children[0], root, 0)
        return node
    def decode(self, data):
        def decodeHelper(root, parent):
            if not root:
                return
            children = []
            node = Node(root.val, children)
            decodeHelper(root.right, node)
            parent.children.append(node)
            decodeHelper(root.left, parent)
        if not data:
            return None
        children = []
        node = Node(data.val, children)
        decodeHelper(data.right, node)
        return node
EOF
def has_cycle(head):
    curr = head
    seen = set()
    while curr:
        if curr in seen:
            return True
        seen.add(curr)
        curr = curr.next
    return False
EOF
def appleAndOrange(s, t, a, b, apple, orange):
    count_app = 0
    count_org = 0
    for el in apple:
        if s <= el+a <= t:
            count_app += 1
    for el in orange:
        if s <= el+b <= t:
            count_org += 1
    return [ count_app, count_org ]
if __name__ == "__main__":
    s, t = input().strip().split(' ')
    s, t = [int(s), int(t)]
    a, b = input().strip().split(' ')
    a, b = [int(a), int(b)]
    m, n = input().strip().split(' ')
    m, n = [int(m), int(n)]
    apple = list(map(int, input().strip().split(' ')))
    orange = list(map(int, input().strip().split(' ')))
    result = appleAndOrange(s, t, a, b, apple, orange)
    print ("\n".join(map(str, result)))
EOF
def _digitSum(number):
    if len(number) == 1:
        return int(number)
    else:
        temp = str(sum([int(x) for x in number]))
        return _digitSum(temp)
def digitSum(number, k):
    temp = str(k*sum([int(x) for x in number]))
    return _digitSum(temp)
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [str(n), int(k)]
    result = digitSum(n, k)
    print(result)
EOF
class Solution(object):
    def depthSum(self, nestedList):
        def depthSumHelper(nestedList, depth):
            res = 0
            for l in nestedList:
                if l.isInteger():
                    res += l.getInteger() * depth
                else:
                    res += depthSumHelper(l.getList(), depth + 1)
            return res
        return depthSumHelper(nestedList, 1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def depthSumInverse(self, nestedList):
        depth_sums = []             
        for nested in nestedList:
            self.dfs(nested, 0, depth_sums)
        total = 0
        max_depth = len(depth_sums)
        for i, depth_sum in enumerate(depth_sums):
            total += (max_depth - i) * depth_sum
        return total
    def dfs(self, nested, depth, depth_sums):
        if len(depth_sums) <= depth:    
            depth_sums.append(0)
        if nested.isInteger():          
            depth_sums[depth] += nested.getInteger()
        else:
            for n in nested.getList():  
                self.dfs(n, depth + 1, depth_sums)
class Solution2(object):
    def depthSumInverse(self, nestedList):
        unweighted, weighted = 0, 0
        q = nestedList
        while q:
            new_q = []
            for nested in q:
                if nested.isInteger():
                    unweighted += nested.getInteger()
                else:
                    new_q += nested.getList()
            q = new_q
            weighted += unweighted
        return weighted
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TrieNode(object):
    def __init__(self):
        self.children = {}      
        self.terminal = False   
class Trie(object):
    def __init__(self):
        self.root = TrieNode()
        self.root.terminal = True   
    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:  
                node.children[c] = TrieNode()
            node = node.children[c]
        node.terminal = True            
    def search(self, word):
        node = self.root
        for c in word:
            if c in node.children:
                node = node.children[c]
            else:
                return False
        return node.terminal            
    def startsWith(self, prefix):
        node = self.root
        for c in prefix:
            if c in node.children:
                node = node.children[c]
            else:
                return False
        return True
EOF
class Solution(object):
    def climbStairs(self, n):
        def matrix_expo(A, K):
            result = [[int(i==j) for j in xrange(len(A))] \
                      for i in xrange(len(A))]
            while K:
                if K % 2:
                    result = matrix_mult(result, A)
                A = matrix_mult(A, A)
                K /= 2
            return result
        def matrix_mult(A, B):
            ZB = zip(*B)
            return [[sum(a*b for a, b in itertools.izip(row, col)) \
                     for col in ZB] for row in A]
        T = [[1, 1],
             [1, 0]]
        return matrix_mult([[1,  0]], matrix_expo(T, n))[0][0]  
class Solution2(object):
    def climbStairs(self, n):
        prev, current = 0, 1
        for i in xrange(n):
            prev, current = current, prev + current,
        return current
EOF
n = int(input().strip())
for _ in range(n):
    tel = input().strip()
    pattern = '^[789][0-9]{9}$'
    print("{}".format("YES" if bool(re.match(pattern, tel)) else "NO"))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def tree2str(self, t):
        result = []
        def preorder(node):
            if not node:
                return
            result.append(str(node.val))
            if not node.left and not node.right:
                return
            result.append("(")
            preorder(node.left)
            result.append(")")
            if node.right:
                result.append("(")
                preorder(node.right)
                result.append(")")
        preorder(t)
        return "".join(result)
EOF
def arrays(arr):
    return numpy.array(arr[::-1], float)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numRescueBoats(self, people, limit):
        boats = 0
        people.sort()
        light, heavy = 0, len(people) - 1
        while light <= heavy:
            boats += 1                                  
            if people[light] + people[heavy] <= limit:  
                light += 1
            heavy -= 1
        return boats
EOF
class Solution(object):
    def removePalindromeSub(self, s):
        def is_palindrome(s):
            for i in xrange(len(s)//2):
                if s[i] != s[-1-i]:
                    return False
            return True
        return 2 - is_palindrome(s) - (s == "")
EOF
def pairs(k, arr):
    res = 0
    memo = dict()
    for el in arr:
        if el-k in memo:
            res += 1
        if el+k in memo:
            res += 1
        memo[el] = True
    return res
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    arr = list(map(int, input().rstrip().split()))
    result = pairs(k, arr)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
def binomial(x, y):
    if y == x:
        return 1
    elif y == 1:         
        return x
    elif y > x:          
        return 0
    else:                
        a = math.factorial(x)
        b = math.factorial(y)
        c = math.factorial(x-y)  
        div = a // (b * c)
        return div
def get_all_substrings(input_string):
    length = len(input_string)
    return [input_string[i:j+1] for i in range(length) for j in range(i,length)]
def sherlockAndAnagrams(s):
    res = 0
    handict = defaultdict(int)
    for sub in get_all_substrings(s):
        handict[str(sorted(sub))] += 1
    for el in list(filter(lambda x: x[1] != 1, handict.items())):
        res += binomial(el[1], 2)
    return res
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = sherlockAndAnagrams(s)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countPrimeSetBits(self, L, R):
        result = 0
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        for i in range(L, R + 1):
            if bin(i).count("1") in primes:
                result += 1
        return result
EOF
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
class Solution(object):
    def postorder(self, root):
        if not root:
            return []
        result, stack = [], [root]
        while stack:
            node = stack.pop()
            result.append(node.val)
            for child in node.children:
                if child:
                    stack.append(child)
        return result[::-1]
class Solution2(object):
    def postorder(self, root):
        def dfs(root, result):
            for child in root.children:
                if child:
                    dfs(child, result)
            result.append(root.val)
        result = []
        if root:
            dfs(root, result)
        return result
EOF
def insertion_sort(l):
    for i in range(1, len(l)):
        j = i-1
        key = l[i]
        while (j >= 0) and (l[j] > key):
            l[j+1] = l[j]
            j -= 1
        l[j+1] = key
m = int(input().strip())
ar = [int(i) for i in input().strip().split()]
insertion_sort(ar)
print(" ".join(map(str,ar)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def encode(self, s, memo = {}):     
        if s in memo:
            return memo[s]
        encodings = [s]                 
        i = (s + s).find(s, 1)          
        if i != -1 and i != len(s):     
            encodings.append(str(len(s) / i) + "[" + self.encode(s[:i], memo) + "]")    
        for i in range(1, len(s)):      
            encodings.append(self.encode(s[:i], memo) + self.encode(s[i:], memo))
        result = min(encodings, key = len)  
        memo[s] = result
        return result
EOF
class Solution(object):
    def integerBreak(self, n):
        if n < 4:
            return n - 1
        res = 0
        if n % 3 == 0:            
            res = 3 ** (n // 3)
        elif n % 3 == 2:          
            res = 3 ** (n // 3) * 2
        else:                     
            res = 3 ** (n // 3 - 1) * 4
        return res
class Solution2(object):
    def integerBreak(self, n):
        if n < 4:
            return n - 1
        res = [0, 1, 2, 3]
        for i in xrange(4, n + 1):
            res[i % 4] = max(res[(i - 2) % 4] * 2, res[(i - 3) % 4] * 3)
        return res[n % 4]
EOF
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def numComponents(self, head, G):
        lookup = set(G)
        dummy = ListNode(-1)
        dummy.next = head
        curr = dummy
        result = 0
        while curr and curr.next:
            if curr.val not in lookup and curr.next.val in lookup:
                result += 1
            curr = curr.next
        return result
EOF
def luckBalance(n, k, arr):
    res = sum(list(map(lambda x: x[0], filter(lambda x: x[1] == 0, arr))))
    arr = sorted(arr, key=lambda x: (-x[1], -x[0]))
    important = len(list(filter(lambda x: x[1] == 1, arr)))
    kcnt = 0
    for ind in range(important):
        if kcnt < k:
            res += arr[ind][0]
            kcnt += 1
        else:
            res -= arr[ind][0]
    return res
if __name__ == "__main__":
    n, k = input().strip().split(' ')
    n, k = [int(n), int(k)]
    contests = []
    for contests_i in range(n):
        contests_t = [int(contests_temp) for contests_temp in input().strip().split(' ')]
        contests.append(contests_t)
    result = luckBalance(n, k, contests)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getRow(self, rowIndex):
        row = [1]
        for i in range(rowIndex):
            row = [1] + [row[i]+row[i+1] for i in range(len(row)-1)] + [1]
        return row
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def getNode(head, positionFromTail):
    cur = head
    for _ in range(positionFromTail):
        cur = cur.next
    res = head
    while cur.next:
        cur = cur.next
        res = res.next
    return res.data
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        position = int(input())
        result = getNode(llist.head, position)
        fptr.write(str(result) + '\n')
    fptr.close()
EOF
symbols_low = string.ascii_lowercase
symbols_up = string.ascii_uppercase
def caesarCipher(s, k):
    res = []
    for c in s:
        if c.isupper():
            res.append(symbols_up[(symbols_up.index(c)+k)%len(symbols_up)])
        elif c.islower():
            res.append(symbols_low[(symbols_low.index(c)+k)%len(symbols_low)])
        else:
            res.append(c)
    return "".join(map(str, res))
if __name__ == "__main__":
    n = int(input().strip())
    s = input().strip()
    k = int(input().strip())
    result = caesarCipher(s, k)
    print(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def distributeCandies(self, candies):
        return min(len(candies) // 2, len(set(candies)))
EOF
t = int(input().strip())
for _ in range(t):
    name, email = input().strip().split()
    if re.match(r'<[A-Za-z](\w|-|\.|_)+        print("{} {}".format(name, email))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def hammingWeight(self, n):
        return sum(c == "1" for c in bin(n)[2:])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findDuplicateSubtrees(self, root):
        def serialize(node):
            if not node:
                return "
            serial = str(node.val) + "," + serialize(node.left) + "," + serialize(node.right)
            subtrees[serial].append(node)
            return serial
        subtrees = defaultdict(list)    
        serialize(root)                 
        return [nodes[0] for serial, nodes in subtrees.items() if len(nodes) > 1]
EOF
def countInversions(array):
    cnt_inv = 0
    if len(array) <= 1:
        return array, 0
    ar_left, inv_left = countInversions(array[:int(len(array)/2)])
    ar_right, inv_right = countInversions(array[int(len(array)/2):])
    ar_merged = []
    i = j = 0
    len_left = len(ar_left)
    len_right = len(ar_right)
    for k in range(len(array) - 1):
        if i == len_left or j == len_right:
            break
        if ar_left[i] <= ar_right[j]:
            ar_merged.append(ar_left[i])
            i += 1
        else:
            ar_merged.append(ar_right[j])
            j += 1
            cnt_inv += len_left - i
    ar_merged += ar_left[i:]
    ar_merged += ar_right[j:]
    return ar_merged, cnt_inv + inv_left + inv_right
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n = int(input().strip())
        arr = list(map(int, input().strip().split(' ')))
        arr_sorted, result = countInversions(arr)
        print(result)
EOF
def isPrime(n):
    if n == 2:
        return True
    elif n == 1:
        return False
    for i in range(2, int(n**.5) + 1):
        if (n % i) == 0:
            return False
    return True
for _ in range(int(input())):
    if isPrime(int(input())):
        print("Prime")
    else:
        print("Not prime")
EOF
def generate_binary(no):
    binary_string = '0' if no % 2 == 0 else '1'
    no = no // 2
    while no != 0:
         if no % 2 == 0:
           binary_string += '0'
           no = no // 2
         else:
            binary_string += '1'
            no = no // 2
    return binary_string
n = int(input().strip())
count = len(max(generate_binary(n).split('0')))
print(count)
EOF
def reverseArray(a):
    return a[::-1]
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr_count = int(input())
    arr = list(map(int, input().rstrip().split()))
    res = reverseArray(arr)
    fptr.write(' '.join(map(str, res)))
    fptr.write('\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def grayCode(self, n):
        gray = [0]
        for i in range(n):
            gray += [x + 2 ** i for x in reversed(gray)]
        return gray
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        start, tank, total = 0, 0, 0
        for station in range(len(gas)):
            balance = gas[station] - cost[station]
            tank += balance
            total += balance
            if tank < 0:
                start = station + 1
                tank = 0
        return -1 if total < 0 else start
EOF
values = [int(i) for i in input().split()]
print("%s %s" % (sum(sorted(values)[:4]), sum(sorted(values)[1:])))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def scheduleCourse(self, courses):
        total_length = 0
        taken_courses = []
        courses.sort(key=lambda c: c[1])  
        for duration, end in courses:
            if total_length + duration <= end:  
                total_length += duration
                heapq.heappush(taken_courses, -duration)
            elif -taken_courses[0] > duration:  
                neg_longest = heapq.heappushpop(taken_courses, -duration)
                total_length += neg_longest + duration
        return len(taken_courses)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def primePalindrome(self, N):
        def is_prime(x):
            if x < 2 or x % 2 == 0:                     
                return x == 2
            for i in range(3, int(x ** 0.5) + 1, 2):    
                if x % i == 0:
                    return False
            return True
        if 8 <= N <= 11:                                
            return 11
        n = len(str(N))
        lhs = 10 ** (n // 2)                            
        while True:
            candidate = int(str(lhs) + str(lhs)[-2::-1])    
            if candidate >= N and is_prime(candidate):
                return candidate
            lhs += 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        mag_count = Counter(magazine)
        ransom_count = Counter(ransomNote)
        for c in ransom_count:
            if c not in mag_count or mag_count[c] < ransom_count[c]:
                return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def majorityElement(self, nums):
        cand1, count1 = None, 0
        cand2, count2 = None, 0
        for num in nums:
            if num == cand1:    
                count1 += 1
            elif num == cand2:  
                count2 += 1
            elif count1 == 0:   
                cand1 = num
                count1 = 1
            elif count2 == 0:   
                cand2 = num
                count2 = 1
            else:       
                count1 -= 1
                count2 -= 1
        return [n for n in (cand1, cand2) if nums.count(n) > len(nums) // 3]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def ambiguousCoordinates(self, S):
        def insert_decimal(s):  
            if s == "0":        
                return [s]
            if s[0] == "0" and s[-1] == "0":    
                return []
            if s[0] == "0":     
                return ["0." + s[1:]]
            if s[-1] == "0":    
                return [s]
            return [s[:i] + "." + s[i:] for i in range(1, len(s))] + [s]
        S = S[1:-1]             
        result = []
        for i in range(1, len(S)):  
            left = insert_decimal(S[:i])
            right = insert_decimal(S[i:])
            result += ["(" + ", ".join([l, r]) + ")" for l in left for r in right]
        return result
EOF
def fun(email):
    pattern = '^[a-zA-Z][\w-]*    return re.match(pattern, email)
EOF
def twoStrings(s1, s2):
    if len(set(s1) & set(s2)) > 0:
        return 'YES'
    else:
        return 'NO'
q = int(input().strip())
for a0 in range(q):
    s1 = input().strip()
    s2 = input().strip()
    result = twoStrings(s1, s2)
    print(result)
EOF
def diagonalDifference(arr):
    sum1 = sum2 = 0
    end = n-1
    for i in range(len(arr)):
        sum1 += arr[i][i]
        sum2 += arr[i][end]
        end -= 1
    return abs(sum1-sum2)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    result = diagonalDifference(arr)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
class Solution(object):
    def minDominoRotations(self, A, B):
        intersect = reduce(set.__and__, [set(d) for d in itertools.izip(A, B)])
        if not intersect:
            return -1
        x = intersect.pop()
        return min(len(A)-A.count(x), len(B)-B.count(x))
EOF
class Solution(object):
    def findMin(self, nums):
        left, right = 0, len(nums)
        target = nums[-1]
        while left < right:
            mid = left + (right - left) / 2
            if nums[mid] <= target:
                right = mid
            else:
                left = mid + 1
        return nums[left]
class Solution2(object):
    def findMin(self, nums):
        left, right = 0, len(nums) - 1
        while left < right and nums[left] >= nums[right]:
            mid = left + (right - left) / 2
            if nums[mid] < nums[left]:
                right = mid
            else:
                left = mid + 1
        return nums[left]
EOF
class Solution(object):
    def numSteps(self, s):
        result, carry = 0, 0
        for i in reversed(xrange(1, len(s))):
            if int(s[i]) + carry == 1:
                carry = 1  
                result += 2
            else:
                result += 1
        return result+carry
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def projectionArea(self, grid):
        n = len(grid)
        row_heights, col_heights = [0] * n, [0] * n
        base_area = 0
        for row in range(n):
            for col in range(n):
                if grid[row][col] != 0:
                    base_area += 1
                row_heights[row] = max(row_heights[row], grid[row][col])
                col_heights[col] = max(col_heights[col], grid[row][col])
        return base_area + sum(row_heights) + sum(col_heights)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        substrings, repeated = set(), set()
        TARGET = 10
        for i in range(len(s)-TARGET+1):
            substring = s[i:i+TARGET]
            if substring in substrings:
                repeated.add(substring)
            else:
                substrings.add(substring)
        return list(repeated)       
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findRadius(self, houses, heaters):
        heaters.sort()
        houses.sort()
        heaters = [float("-inf")] + heaters + [float("inf")]
        i = 0
        radius = -1
        for house in houses:
            while heaters[i + 1] < house:
                i += 1
            left_distance = house - heaters[i]
            right_distance = heaters[i + 1] - house         
            closest = min(left_distance, right_distance)
            radius = max(radius, closest)
        return radius
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def validWordSquare(self, words):
        if len(words) != len(words[0]):     
            return False
        n = len(words[0])
        for i, word in enumerate(words[1:], 1):
            m = len(word)
            if m > n:                       
                return False
            words[i] += (n - m) * " "       
        for i in range(n):
            for j in range(n):
                if words[i][j] != words[j][i]:
                    return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def consecutiveNumbersSum(self, N):
        k = 1                               
        temp = N - ((k + 1) * k) // 2       
        result = 0
        while temp >= 0:
            if temp % k == 0:               
                result += 1
            k += 1
            temp = N - ((k + 1) * k) // 2
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProfit(self, k, prices):
        if k >= len(prices) // 2:       
            return sum([max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))])
        buys, sells = [float('-inf') for _ in range(k + 1)], [0 for _ in range(k + 1)]
        for price in prices:
            for i in range(1, len(buys)):
                buys[i] = max(buys[i], sells[i-1] - price)  
                if buys[i] == buys[i-1]:                    
                    break
                sells[i] = max(sells[i], buys[i] + price)   
        return max(sells)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def pruneTree(self, root):
        def contains_one(node):
            if not node:
                return False
            left_one, right_one = contains_one(node.left), contains_one(node.right) 
            if not left_one:                                
                node.left = None
            if not right_one:
                node.right = None
            return node.val == 1 or left_one or right_one   
        return root if contains_one(root) else None         
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def fizzBuzz(self, n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numDecodings(self, s):
        ways = 0
        if s[0] == "*":
            ways = 9
        elif s[0] != "0":
            ways = 1
        prev_char = s[0]
        prev_ways = 1
        for c in s[1:]:
            new = 0
            if c == "*":
                new = 9 * ways
            elif c != "0":
                new = ways
            if prev_char == "*":
                if c == "*":                
                    new += prev_ways * 15
                elif "0" <= c <= "6":       
                    new += prev_ways * 2
                elif "7" <= c <= "9":       
                    new += prev_ways
            elif prev_char == "1":
                if c == "*":
                    new += prev_ways * 9
                else:
                    new += prev_ways
            elif prev_char == "2":
                if c == "*":
                    new += prev_ways * 6
                elif c <= "6":
                    new += prev_ways
            new %= 10 ** 9 + 7
            prev_ways, ways = ways, new
            prev_char = c
        return ways
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProduct(self, words):
        codes = []              
        for word in words:
            codes.append(sum(1 << (ord(c) - ord('a')) for c in set(word)))
        max_product = 0
        for i in range(len(codes)-1):
            for j in range(i+1, len(codes)):
                if not (codes[i] & codes[j]):
                    max_product = max(max_product, len(words[i]) * len(words[j]))
        return max_product
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def sortedArrayToBST(self, nums):
        return self.convert(nums, 0, len(nums)-1)
    def convert(self, nums, left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = self.convert(nums, left, mid-1)
        root.right = self.convert(nums, mid+1, right)
        return root
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def validTicTacToe(self, board):
        counts, lines = [0, 0], [0, 0]          
        for i, char in enumerate(("O", "X")):
            for j, row in enumerate(board):
                if row == char * 3:             
                    lines[i] += 1
                if board[0][j] == board[1][j] == board[2][j] == char:   
                    lines[i] += 1
                for c in row:                   
                    if c == char:
                        counts[i] += 1
            if board[0][0] == board[1][1] == board[2][2] == char:       
                lines[i] += 1
            if board[2][0] == board[1][1] == board[0][2] == char:       
                lines[i] += 1
        if lines[0] and lines[1]:                                       
            return False
        if lines[0] and counts[0] != counts[1]:                         
            return False
        if lines[1] and counts[1] != counts[0] + 1:                     
            return False
        if counts[1] - counts[0] > 1 or counts[1] - counts[0] < 0:      
            return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def solveNQueens(self, n):
        partials = [[]]                         
        for col in range(n):
            new_partials = []
            for partial in partials:
                for row in range(n):
                    if not self.conflict(partial, row):
                        new_partials.append(partial + [row])
            partials = new_partials
        results = []
        for partial in partials:                
            result = [['.'] * n for _ in range(n)]
            for col, row in enumerate(partial):
                result[row][col] = 'Q'
            for row in range(n):
                result[row] = ''.join(result[row])
            results.append(result)
        return results
    def conflict(self, partial, new_row):
        for col, row in enumerate(partial):
            if new_row == row:                      
                return True
            col_diff = len(partial) - col
            if abs(new_row - row) == col_diff:      
                return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def closestValue(self, root, target):
        closest = root.val
        while root:
            if root.val == target:      
                return root.val
            if abs(root.val - target) < abs(closest - target):
                closest = root.val
            if target < root.val:
                root = root.left
            else:
                root = root.right
        return closest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def kthSmallest(self, root, k):
        stack = []
        while root:
            stack.append(root)
            root = root.left
        while stack:
            node = stack.pop()
            k -= 1
            if k == 0:
                return node.val
            node = node.right
            while node:
                stack.append(node)
                node = node.left
class Solution2(object):
    def kthSmallest(self, root, k):
        self.k = k              
        self.result = None
        self.helper(root)
        return self.result
    def helper(self, node):
        if not node:
            return
        self.helper(node.left)
        self.k -= 1
        if self.k == 0:
            self.result = node.val
            return
        self.helper(node.right)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def reverseBits(self, n):
        binary = bin(n)
        binary = binary[2:]             
        reversed_binary = binary[::-1] + ''.join(['0' for _ in range(32 - len(binary))])
        return int(reversed_binary, 2)  
class Solution2:
    def reverseBits(self, n):
        reversed, bit = 0, 31
        while n != 0:
            if n % 2 == 1:      
                reversed += 2**bit
            bit -= 1
            n //= 2
        return reversed
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findSubstring(self, s, words):
        result = []
        word_len = len(words[0])
        for stripe in range(word_len):  
            i = stripe                  
            to_match = len(words)       
            freq = Counter(words)       
            while i + to_match*word_len <= len(s):  
                word = s[i:i+word_len]   
                if word in freq:         
                    freq[word] -= 1
                    if freq[word] == 0:
                        del freq[word]
                    to_match -= 1
                    i += word_len
                    if to_match == 0:               
                        result.append(i - word_len*len(words))
                elif to_match != len(words):        
                    nb_matches = len(words) - to_match
                    first_word = s[i - nb_matches*word_len:i - (nb_matches-1)*word_len]
                    freq.setdefault(first_word, 0)  
                    freq[first_word] += 1
                    to_match += 1
                else:                               
                    i += word_len
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, N, blacklist):
        self.white = N - len(blacklist)         
        blacklist = set(blacklist)
        self.white_to_move = [i for i in range(self.white, N) if i not in blacklist]
        self.mapping = {b : self.white_to_move.pop() for b in blacklist if b < self.white}
    def pick(self):
        rand = randint(0, self.white - 1)
        return self.mapping[rand] if rand in self.mapping else rand
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def nearestPalindromic(self, n):
        digits = len(n)
        candidates = {int("1" + "0" * (digits - 1) + "1")}  
        if len(n) > 1:
            candidates.add(int("9" * (digits - 1)))  
        mid = len(n) // 2  
        left = n[:mid]
        if len(n) % 2 == 1:
            centre_count = 1  
            centre = n[mid]
            right = left[::-1]
        else:
            centre_count = 2  
            centre = left[-1]
            left = left[:-1]
            right = left[::-1]
        candidates.add(int(left + centre * centre_count + right))
        if centre != "9":
            new_centre = str(int(centre) + 1)
            candidates.add(int(left + new_centre * centre_count + right))
        if centre != "0":
            new_centre = str(int(centre) - 1)
            candidates.add(int(left + new_centre * centre_count + right))
        n_int = int(n)
        candidates.discard(n_int)
        candidates = list(candidates)
        candidates.sort(key=lambda x: (abs(x - n_int), x))  
        return str(candidates[0])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def flatten(self, head):
        node = head                     
        while node:                     
            if node.child:
                old_next = node.next    
                node.next = self.flatten(node.child)    
                node.next.prev = node   
                node.child = None       
                while node.next:
                    node = node.next    
                node.next = old_next    
                if old_next:            
                    old_next.prev = node
            node = node.next            
        return head
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canWinNim(self, n):
        return n % 4 != 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def validPalindrome(self, s):
        n = len(s)
        i = 0
        while i < n // 2:
            if s[i] != s[n - 1 - i]:
                del_front = s[i + 1:n - i]
                del_back = s[i:n - 1 - i]
                return del_front == del_front[::-1] or del_back == del_back[::-1]
            i += 1
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def setZeroes(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        rows = len(matrix)
        cols = len(matrix[0])
        first_row_zero = any([True for c in range(cols) if matrix[0][c] == 0])
        first_col_zero = any([True for r in range(rows) if matrix[r][0] == 0])
        for r in range(1, rows):
            for c in range(1, cols):
                if matrix[r][c] == 0:
                    matrix[r][0] = 0
                    matrix[0][c] = 0
        for r in range(1, rows):
            if matrix[r][0] == 0:
                for c in range(1, cols):
                    matrix[r][c] = 0
        for c in range(1, cols):
            if matrix[0][c] == 0:
                for r in range(1, rows):
                    matrix[r][c] = 0
        if first_row_zero:
            matrix[0] = [0] * cols
        if first_col_zero:
            for r in range(rows):
                matrix[r][0] = 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    memo = [0, 1]   
    def numSquares(self, n):
        while len(self.memo) <= n:
            self.memo.append(1 + min(self.memo[-i*i] for i in range(1, int(len(self.memo)**0.5)+1)))
        return self.memo[n]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def thirdMax(self, nums):
        maxima = [float("-inf")] * 3    
        for num in nums:
            if num in maxima:
                continue
            if num > maxima[0]:
                maxima = [num] + maxima[:2]
            elif num > maxima[1]:
                maxima[1:] = [num, maxima[1]]
            elif num > maxima[2]:
                maxima[2] = num
        return maxima[2] if maxima[2] != float("-inf") else maxima[0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numDistinct(self, s, t):
        prev_subsequences = [1 for _ in range(len(s) + 1)]    
        for r in range(1, len(t) + 1):                        
            subsequences = [0 for _ in range(len(s) + 1)]
            for c in range(r, len(s) + 1):                    
                subsequences[c] = subsequences[c - 1]         
                if s[c - 1] == t[r - 1]:                      
                    subsequences[c] += prev_subsequences[c - 1]
            prev_subsequences = subsequences
        return prev_subsequences[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def deleteAndEarn(self, nums):
        freq = Counter(nums)
        pairs = [(num, count) for num, count in freq.items()]
        pairs.sort()
        used, not_used = 0, 0
        for i, (num, count) in enumerate(pairs):
            if i == 0 or pairs[i - 1][0] != num - 1:    
                not_used = max(used, not_used)          
                used = num * count + not_used           
            else:
                used, not_used = num * count + not_used, max(used, not_used)
        return max(used, not_used)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shoppingOffers(self, price, special, needs):
        def helper():
            needs_tuple = tuple(needs)
            if needs_tuple in memo:
                return memo[needs_tuple]
            min_cost = 0
            for cost, need in zip(price, needs):
                min_cost += need * cost
            if min_cost == 0:       
                return 0
            for offer in special:
                for i, need in enumerate(needs):
                    if offer[i] > need:
                        break       
                else:
                    for i, need in enumerate(needs):
                        needs[i] -= offer[i]
                    min_cost = min(min_cost, offer[-1] + helper())
                    for i, need in enumerate(needs):
                        needs[i] += offer[i]
            memo[needs_tuple] = min_cost
            return min_cost
        memo = {}
        return helper()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def hitBricks(self, grid, hits):
        rows, cols = len(grid), len(grid[0])
        nbors = ((1, 0), (0, 1), (-1, 0), (0, -1))
        for r, c in hits:  
            grid[r][c] -= 1
        def dfs(row, col):
            if row < 0 or row >= rows or col < 0 or col >= cols:
                return 0
            if grid[row][col] != 1:
                return 0
            grid[row][col] = 2
            return 1 + sum(dfs(row + dr, col + dc) for dr, dc in nbors)
        for c in range(cols):
            dfs(0, c)
        def connected(r, c):
            if r == 0:
                return True
            return any(0 <= (r + dr) < rows and 0 <= (c + dc) < cols \
                       and grid[r + dr][c + dc] == 2 for dr, dc in nbors)
        result = []
        for r, c in reversed(hits):
            grid[r][c] += 1
            if grid[r][c] == 1 and connected(r, c):
                result.append(dfs(r, c) - 1)  
            else:
                result.append(0)
        return result[::-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minimumTotal(self, triangle):
        for row in range(len(triangle)-2, -1, -1):
            for col in range(len(triangle[row])):
                triangle[row][col] += min(triangle[row+1][col], triangle[row+1][col+1])
        return triangle[0][0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def deckRevealedIncreasing(self, deck):
        n = len(deck)
        index = deque(range(n))
        result = [None] * n
        for card in sorted(deck):
            result[index.popleft()] = card
            if index:
                index.append(index.popleft())
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def flipLights(self, n, m):
        if n == 0:
            return 0        
        if m == 0:
            return 1        
        if m == 1:          
            if n == 1:
                return 2    
            if n == 2:
                return 3    
            return 4        
        if m == 2:          
            if n == 1:
                return 2    
            if n == 2:
                return 4    
            if n >= 3:
                return 7    
        return 2 ** min(n, 3)  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isIsomorphic(self, s, t):
        if len(s) != len(t):
            return False
        s_to_t = {}
        t_mapped = set()
        for cs, ct in zip(s, t):
            if cs in s_to_t:
                if s_to_t[cs] != ct:
                    return False
            elif ct in t_mapped:
                return False
            s_to_t[cs] = ct
            t_mapped.add(ct)
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class RandomizedSet(object):
    def __init__(self):
        self.mapping = {}   
        self.items = []     
    def insert(self, val):
        if val not in self.mapping:  
            self.items.append(val)
            self.mapping[val] = len(self.items) - 1
            return True
        return False
    def remove(self, val):
        if val not in self.mapping:
            return False
        index = self.mapping[val]
        self.items[index] = self.items[-1]  
        self.mapping[self.items[index]] = index  
        self.items.pop()
        del self.mapping[val]
        return True
    def getRandom(self):
        return self.items[random.randint(0, len(self.items) - 1)]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numSquarefulPerms(self, A):
        freq = Counter(A)               
        pairs = defaultdict(set)        
        unique = list(freq.keys())
        pairs[None] = unique            
        for i, num1 in enumerate(unique):   
            for num2 in unique[i:]:
                if int((num1 + num2) ** 0.5) ** 2 == num1 + num2:
                    pairs[num1].add(num2)
                    pairs[num2].add(num1)
        def helper(num, length):        
            if length == len(A):        
                return 1
            count = 0
            for next_num in pairs[num]: 
                if freq[next_num] > 0:  
                    freq[next_num] -= 1 
                    count += helper(next_num, length + 1)
                    freq[next_num] += 1 
            return count
        return helper(None, 0)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, radius, x_center, y_center):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center
    def randPoint(self):
        x, y = 2 * random.random() - 1, 2 * random.random() - 1     
        if x * x + y * y > 1:                                       
            return self.randPoint()
        return [x * self.radius + self.x_center, y * self.radius + self.y_center]   
    def randPoint2(self):
        radius = random.random() ** 0.5 * self.radius
        angle = random.random() * 2 * math.pi
        return [radius * math.sin(angle) + self.x_center, radius * math.cos(angle) + self.y_center]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def isCompleteTree(self, root):
        queue = deque([root])
        while True:
            node = queue.popleft()
            if not node:
                return all(not nd for nd in queue)
            queue.append(node.left)
            queue.append(node.right)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def validSquare(self, p1, p2, p3, p4):
        def square_dist(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        points = [p1, p2, p3, p4]
        square_dists = [square_dist(points[i], points[j]) for i in range(4) for j in range(i + 1, 4)]
        side = min(square_dists)
        if max(square_dists) != 2 * side:
            return False
        return square_dists.count(side) == 2 * square_dists.count(2 * side)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def intToRoman(self, num):
        mapping = [(1000, 'M'),
                    (900, 'CM'),
                    (500, 'D'),
                    (400, 'CD'),
                    (100, 'C'),
                    (90, 'XC'),
                    (50, 'L'),
                    (40, 'XL'),
                    (10, 'X'),
                    (9, 'IX'),
                    (5, 'V'),
                    (4, 'IV'),
                    (1, 'I'),]
        roman = []
        for i, numeral in mapping:
            while num >= i:
                num -= i
                roman.append(numeral)
        return "".join(roman)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestCommonPrefix(self, strs):
        if not strs:
            return ''
        strs.sort()
        first = strs[0]
        last = strs[-1]
        i = 0
        while i < len(first) and i < len(last) and first[i] == last[i]:
            i += 1
        return first[:i]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def kthSmallestPrimeFraction(self, A, K):
        def count_smaller_fractions(x):     
            count, denominator, largest = 0, 1, [0, 1]
            for numerator in range(len(A) - 1):     
                while denominator < len(A) and A[numerator] >= x * A[denominator]:  
                    denominator += 1
                if denominator != len(A) and A[numerator] * largest[1] > largest[0] * A[denominator]:
                    largest = [A[numerator], A[denominator]]                        
                count += len(A) - denominator       
            return count, largest
        low, high = 0, 1.0
        while high - low > 10 ** -9:
            mid = (low + high) / 2
            count, largest = count_smaller_fractions(mid)
            if count < K:           
                low = mid
            else:                   
                result = largest    
                high = mid
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minMoves2(self, nums):
        nums.sort()
        front, back = 0, len(nums) - 1
        moves = 0
        while front < back:
            moves += nums[back] - nums[front]
            front += 1
            back -= 1
        return moves
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class RangeModule(object):
    def __init__(self):
        self.points = [0, 10 ** 9]  
        self.in_range = [False, False]  
    def addRange(self, left, right, add=True):
        i = bisect.bisect_left(self.points, left)  
        if self.points[i] != left:  
            self.points.insert(i, left)  
            self.in_range.insert(i, self.in_range[i - 1])  
        j = bisect.bisect_left(self.points, right)
        if self.points[j] != right:
            self.points.insert(j, right)
            self.in_range.insert(j, self.in_range[j - 1])  
        self.points[i:j] = [left]  
        self.in_range[i:j] = [add]  
    def queryRange(self, left, right):
        i = bisect.bisect(self.points, left) - 1    
        j = bisect.bisect_left(self.points, right)  
        return all(self.in_range[i:j])
    def removeRange(self, left, right):
        self.addRange(left, right, False)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def pancakeSort(self, A):
        flips = []
        for unsorted in range(len(A), 0, -1):       
            i = A.index(unsorted)
            if i == unsorted - 1:                   
                continue
            A = A[unsorted - 1:i:-1] + A[:i + 1] + A[unsorted:]
            flips += [i + 1, unsorted]
        return flips
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def trailingZeroes(self, n):
        zeroes = 0
        power_of_5 = 5
        while power_of_5 <= n:
            zeroes += n // power_of_5
            power_of_5 *= 5
        return zeroes
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def slidingPuzzle(self, board):
        nbors = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]  
        def next_boards(b):                 
            i = b.index(0)
            next_bds = []
            for nbor in nbors[i]:
                b_copy = b[:]               
                b_copy[i], b_copy[nbor] = b_copy[nbor], b_copy[i]   
                next_bds.append(b_copy)
            return next_bds
        queue = [board[0] + board[1]]       
        steps = 0
        seen = set()                        
        while queue:
            new_queue = []
            for bd in queue:
                if bd == [1, 2, 3, 4, 5, 0]:
                    return steps
                seen.add(tuple(bd))
                new_queue += [nb for nb in next_boards(bd) if tuple(nb) not in seen]
            steps += 1
            queue = new_queue
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def totalHammingDistance(self, nums):
        n = len(nums)
        hamming = 0
        for bit in range(32):
            set_bits = 0  
            for num in nums:
                set_bits += (num >> bit) & 1
            hamming += (n - set_bits) * set_bits
        return hamming
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def smallestFactorization(self, a):
        if a == 1:
            return 1
        result = 0
        tens = 1
        for digit in range(9, 1, -1):
            while a != 1 and a % digit == 0:
                result += digit * tens
                if result > 2 ** 31:
                    return 0
                a //= digit
                if a == 1:
                    return result
                tens *= 10
        return 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestPalindrome(self, s):
        longest_prefix_suffix = self.kmp_table(s + '*' + s[::-1])
        return s[:longest_prefix_suffix:-1] + s
    def kmp_table(self, word):
        failure = [-1] + [0 for _ in range(len(word)-1)]
        pos = 2             
        candidate = 0
        while pos < len(word):
            if word[pos-1] == word[candidate]:  
                failure[pos] = candidate + 1
                candidate += 1
                pos += 1
            elif candidate > 0:                 
                candidate = failure[candidate]
                failure[pos] = 0
            else:   
                failure[pos] = 0
                pos += 1
        return failure[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def snakesAndLadders(self, board):
        linear = [-1]               
        reverse = False             
        for row in board[::-1]:     
            linear += row[::-1] if reverse else row
            reverse = not reverse
        moves = 0
        visited = set()             
        queue = {1}                 
        while queue:
            new_queue = set()
            for i in queue:
                if i in visited or i >= len(linear):    
                    continue
                visited.add(i)
                if linear[i] != -1:                     
                    i = linear[i]
                if i == len(linear) - 1:
                    return moves
                for step in range(1, 7):                
                    new_queue.add(i + step)
            moves += 1
            visited |= queue
            queue = new_queue
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def calcEquation(self, equations, values, queries):
        graph = defaultdict(dict)           
        for i in range(len(equations)):     
            num, den, val = equations[i][0], equations[i][1], values[i]
            graph[num][den] = val
            graph[den][num] = 1 / val
        for i in graph:
            for j in graph[i]:
                for k in graph[i]:
                    graph[j][k] = graph[j][i] * graph[i][k]     
        results = []
        for num, den in queries:
            if num in graph and den in graph[num]:
                results.append(graph[num][den])
            else:
                results.append(-1)
        return results
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class PeekingIterator(object):
    def __init__(self, iterator):
        self.front = None
        self.it = iterator
        if self.it.hasNext():
            self.front = self.it.next()
    def peek(self):
        return self.front   
    def next(self):
        temp = self.front
        self.front = None
        if self.it.hasNext():   
            self.front = self.it.next()
        return temp
    def hasNext(self):
        return bool(self.front)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
def read4(buf):
    pass
class Solution(object):
    def __init__(self):
        self.leftover = deque()     
    def read(self, buf, n):
        total_chars, added_chars, read_chars = 0, 4, 0
        while self.leftover and total_chars < n:        
            buf[total_chars] = self.leftover.popleft()
            total_chars += 1
        while added_chars == 4 and total_chars < n:     
            buf4 = [""] * 4     
            read_chars = read4(buf4)
            added_chars = min(read_chars, n - total_chars)
            buf[total_chars:total_chars+added_chars] = buf4
            total_chars += added_chars
        while read_chars > added_chars:                 
            self.leftover.append(buf4[added_chars])
            added_chars += 1
        return total_chars
EOF
_author_ = 'jake'
_project_ = 'leetcode'
def knows(a, b):
    return
class Solution(object):
    def findCelebrity(self, n):
        candidate = 0
        for i in range(1, n):
            if knows(candidate, i):
                candidate = i
        for i in range(n):
            if i == candidate:
                continue
            if not knows(i, candidate) or knows(candidate, i):
                return -1
        return candidate
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numFriendRequests(self, ages):
        freq = Counter(ages)
        age_counts = [(k, v) for k, v in freq.items()]
        age_counts.sort()
        requests = 0
        for a, (age_a, count_a) in enumerate(age_counts):
            for age_b, count_b in age_counts[:a]:       
                if age_b > 0.5 * age_a + 7 and (age_b < 100 or age_a > 100):
                    requests += count_a * count_b
            if age_a > 14:                              
                requests += count_a * (count_a - 1)
        return requests
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxDistToClosest(self, seats):
        empty_seats = []
        max_distance = 0
        last_seat = float("-inf")
        for i, seat in enumerate(seats):
            if seat == 1:                                   
                while empty_seats:
                    seat_i, left_distance = empty_seats.pop()
                    max_distance = max(max_distance, min(left_distance, i - seat_i))
                last_seat = i
            elif i - last_seat > max_distance:              
                empty_seats.append((i, i - last_seat))
        while empty_seats:                                  
            seat_i, left_distance = empty_seats.pop()
            max_distance = max(max_distance, left_distance)
        return max_distance
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.smaller = 0    
        self.left = None
        self.right = None
class Solution(object):
    def countSmaller(self, nums):
        smaller = [0 for _ in range(len(nums))]
        if len(nums) < 2:
            return smaller
        root = TreeNode(nums[-1])
        for i in range(len(nums)-2, -1, -1):    
            node = root
            count = 0                           
            while True:
                if nums[i] < node.val:
                    node.smaller += 1           
                    if not node.left:
                        node.left = TreeNode(nums[i])
                        break
                    else:
                        node = node.left
                else:   
                    count += node.smaller       
                    if nums[i] > node.val:      
                        count += 1
                    if not node.right:
                        node.right = TreeNode(nums[i])
                        break
                    else:
                        node = node.right
            smaller[i] = count
        return smaller
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rob(self, nums):
        if not nums:
            return 0
        loot, prev = nums[0], 0     
        for num in nums[1:]:
            loot, prev = max(num + prev, loot), loot
        return loot
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        q = deque()
        max_window = []
        for i, num in enumerate(nums):
            while q and nums[q[-1]] < num:  
                q.pop()
            q.append(i)         
            if q[0] <= i - k:   
                q.popleft()
            if i >= k - 1:      
                max_window.append(nums[q[0]])
        return max_window
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestRectangleArea(self, heights):
        max_area = 0
        heights = [0] + heights + [0]   
        stack = [0]                     
        for i, bar in enumerate(heights[1:], 1):
            while heights[stack[-1]] > bar:     
                height = heights[stack.pop()]   
                width = i - stack[-1] - 1       
                max_area = max(max_area, height * width)
            stack.append(i)
        return max_area
EOF
class Solution(object):
    def orangesRotting(self, grid):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        count = 0
        q = collections.deque()
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == 2:
                    q.append((r, c, 0))
                elif val == 1:
                    count += 1
        result = 0
        while q:
            r, c, result = q.popleft()
            for d in directions:
                nr, nc = r+d[0], c+d[1]
                if not (0 <= nr < len(grid) and \
                        0 <= nc < len(grid[r])):
                    continue
                if grid[nr][nc] == 1:
                    count -= 1
                    grid[nr][nc] = 2
                    q.append((nr, nc, result+1))
        return result if count == 0 else -1
EOF
class Solution(object):
    def findRelativeRanks(self, nums):
        sorted_nums = sorted(nums)[::-1]
        ranks = ["Gold Medal", "Silver Medal", "Bronze Medal"] + map(str, range(4, len(nums) + 1))
        return map(dict(zip(sorted_nums, ranks)).get, nums)
EOF
class Solution(object):
    def getMaximumGold(self, grid):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        def backtracking(grid, i, j):
            result = 0
            grid[i][j] *= -1
            for dx, dy in directions:
                ni, nj = i+dx, j+dy
                if not (0 <= ni < len(grid) and
                        0 <= nj < len(grid[0]) and
                        grid[ni][nj] > 0):
                    continue
                result = max(result, backtracking(grid, ni, nj))
            grid[i][j] *= -1
            return grid[i][j] + result
        result = 0
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j]:
                    result = max(result, backtracking(grid, i, j))
        return result
EOF
class Node:
    def __init__(self, info): 
        self.info = info  
        self.left = None  
        self.right = None 
        self.level = None 
    def __str__(self):
        return str(self.info) 
class BinarySearchTree:
    def __init__(self): 
        self.root = None
    def create(self, val):  
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break
def preOrder(root):
    if not root:
        return None
    print(root.info,end=" ")
    preOrder(root.left)
    preOrder(root.right)
tree = BinarySearchTree()
t = int(input())
arr = list(map(int, input().split()))
for i in range(t):
    tree.create(arr[i])
preOrder(tree.root)
EOF
def aVeryBigSum(n, ar):
    sum = 0
    for elem in ar:
        sum += elem
    return sum
n = int(input().strip())
ar = list(map(int, input().strip().split(' ')))
result = aVeryBigSum(n, ar)
print(result)
EOF
if __name__ == "__main__":
    date = list(map(int, input().strip().split(' ')))
    print(calendar.day_name[calendar.weekday(date[2], date[0], date[1])].upper())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minSubArrayLen(self, s, nums):
        subarray_sum, min_length, start = 0, len(nums) + 1, 0   
        for i in range(len(nums)):
            subarray_sum += nums[i]     
            while subarray_sum >= s:    
                min_length = min(min_length, i - start + 1)
                subarray_sum -= nums[start]
                start += 1
        return 0 if min_length > len(nums) else min_length
EOF
def libraryFine(d1, m1, y1, d2, m2, y2):
    fine = 0
    if y1 > y2:
        fine = 10000
    elif m1 > m2 and y1 == y2:
        fine = 500 * (m1 - m2)
    elif d1 > d2 and m1 == m2 and y1 == y2:
        fine = 15 * (d1 - d2)
    return fine
if __name__ == "__main__":
    d1, m1, y1 = input().strip().split(' ')
    d1, m1, y1 = [int(d1), int(m1), int(y1)]
    d2, m2, y2 = input().strip().split(' ')
    d2, m2, y2 = [int(d2), int(m2), int(y2)]
    result = libraryFine(d1, m1, y1, d2, m2, y2)
    print(result)
EOF
def countingValleys(n, s):
    valley,pos = 0, 0
    for i in s:
        if i == "U":
            pos += 1
        elif i == "D":
            pos -= 1
        if pos == 0 and i == "U":
            valley += 1
    return valley
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    s = input()
    result = countingValleys(n, s)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def judgePoint24(self, nums):
        n = len(nums)
        if n == 1:
            return abs(nums[0] - 24) < 0.001
        for i in range(n - 1):
            for j in range(i + 1, n):
                remainder = nums[:i] + nums[i + 1:j] + nums[j + 1:]
                if self.judgePoint24(remainder + [nums[i] + nums[j]]):
                    return True
                if self.judgePoint24(remainder + [nums[i] - nums[j]]):
                    return True
                if self.judgePoint24(remainder + [nums[j] - nums[i]]):
                    return True
                if self.judgePoint24(remainder + [nums[i] * nums[j]]):
                    return True
                if nums[j] != 0 and self.judgePoint24(remainder + [float(nums[i]) / float(nums[j])]):
                    return True
                if nums[i] != 0 and self.judgePoint24(remainder + [float(nums[j]) / float(nums[i])]):
                    return True
        return False
EOF
class UnionFind(object):
    def __init__(self, n):
        self.set = range(n+1)
        self.size = [1]*(n+1)
        self.size[-1] = 0
    def find_set(self, x):
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])  
        return self.set[x]
    def union_set(self, x, y):
        x_root, y_root = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[min(x_root, y_root)] = max(x_root, y_root)
        self.size[max(x_root, y_root)] += self.size[min(x_root, y_root)]
        return True
    def top(self):
        return self.size[self.find_set(len(self.size)-1)]
class Solution(object):
    def hitBricks(self, grid, hits):
        def index(C, r, c):
            return r*C+c
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        R, C = len(grid), len(grid[0])
        hit_grid = [row[:] for row in grid]
        for i, j in hits:
            hit_grid[i][j] = 0
        union_find = UnionFind(R*C)
        for r, row in enumerate(hit_grid):
            for c, val in enumerate(row):
                if not val:
                    continue
                if r == 0:
                    union_find.union_set(index(C, r, c), R*C)
                if r and hit_grid[r-1][c]:
                    union_find.union_set(index(C, r, c), index(C, r-1, c))
                if c and hit_grid[r][c-1]:
                    union_find.union_set(index(C, r, c), index(C, r, c-1))
        result = []
        for r, c in reversed(hits):
            prev_roof = union_find.top()
            if grid[r][c] == 0:
                result.append(0)
                continue
            for d in directions:
                nr, nc = (r+d[0], c+d[1])
                if 0 <= nr < R and 0 <= nc < C and hit_grid[nr][nc]:
                    union_find.union_set(index(C, r, c), index(C, nr, nc))
            if r == 0:
                union_find.union_set(index(C, r, c), R*C)
            hit_grid[r][c] = 1
            result.append(max(0, union_find.top()-prev_roof-1))
        return result[::-1]
EOF
class Solution(object):
    def closedIsland(self, grid):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        def fill(grid, i, j):
            if not (0 <= i < len(grid) and 
                    0 <= j < len(grid[0]) and 
                    grid[i][j] == 0):
                return False
            grid[i][j] = 1
            for dx, dy in directions:
                fill(grid, i+dx, j+dy)
            return True
        for j in xrange(len(grid[0])):
            fill(grid, 0, j)
            fill(grid, len(grid)-1, j)
        for i in xrange(1, len(grid)):
            fill(grid, i, 0)
            fill(grid, i, len(grid[0])-1)
        result = 0
        for i in xrange(1, len(grid)-1):
            for j in xrange(1, len(grid[0])-1):
                if fill(grid, i, j):
                    result += 1
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def __init__(self, head):
        self.head = head
        self.count = 0
        while head:
            self.count += 1
            head = head.next
    def getRandom(self):
        randnode = random.randint(0, self.count - 1)
        node = self.head
        for _ in range(randnode):
            node = node.next
        return node.val
EOF
class Solution(object):
    def maxProfitAssignment(self, difficulty, profit, worker):
        jobs = zip(difficulty, profit)
        jobs.sort()
        worker.sort()
        result, i, max_profit = 0, 0, 0
        for ability in worker:
            while i < len(jobs) and jobs[i][0] <= ability:
                max_profit = max(max_profit, jobs[i][1])
                i += 1
            result += max_profit
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        last_seen = lower-1
        nums.append(upper+1)
        missing = []
        for num in nums:
            if num == last_seen + 2:
                missing.append(str(last_seen+1))
            elif num > last_seen + 2:
                missing.append(str(last_seen+1) + '->' + str(num-1))
            last_seen = num
        return missing
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def makeLargestSpecial(self, S):
        specials = []
        if not S:
            return ""
        balance, start = 0, 0
        for i, c in enumerate(S):
            balance += 1 if c == "1" else -1
            if balance == 0:
                specials.append("1" + self.makeLargestSpecial(S[start + 1:i]) + "0")
                start = i + 1
        specials.sort(reverse = True)
        return "".join(specials)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def expressiveWords(self, S, words):
        Groups = namedtuple("groups", ["chars", "counts"])      
        def get_groups(word):                                   
            groups = Groups(chars = [], counts = [])
            count = 1
            for i, c in enumerate(word):
                if i == len(word) - 1 or c != word[i + 1]:
                    groups.chars.append(c)
                    groups.counts.append(count)
                    count = 1
                else:
                    count += 1
            return groups
        result = 0
        S_groups = get_groups(S)
        for word in words:
            word_groups = get_groups(word)
            if word_groups.chars != S_groups.chars:         
                continue
            for S_count, word_count in zip(S_groups.counts, word_groups.counts):
                if word_count > S_count:                    
                    break
                if word_count < S_count and S_count == 2:   
                    break
            else:
                result += 1
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def depthSum(self, nestedList):
        def helper(nested, depth):
            total = 0
            for item in nested:
                if item.isInteger():
                    total += depth * item.getInteger()
                else:
                    total += helper(item.getList(), depth + 1)
            return total
        return helper(nestedList, 1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isSymmetric(self, root):
        if not root:
            return True
        return self.is_mirror(root.left, root.right)
    def is_mirror(self, left_node, right_node):
        if not left_node and not right_node:
             return True
        if not left_node or not right_node:
             return False
        if left_node.val != right_node.val:
            return False
        return self.is_mirror(right_node.right, left_node.left) and \
               self.is_mirror(left_node.right, right_node.left)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def sumOfLeftLeaves(self, root):
        if not root:
            return 0
        if root.left and not root.left.left and not root.left.right:
            return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyHashMap(object):
    def __init__(self):
        self.size = 10000
        self.hashmap = [[] for _ in range(self.size)]
    def put(self, key, value):
        bucket, index = self.key_index(key)
        if index == -1:
            self.hashmap[bucket].append([key, value])
        else:
            self.hashmap[bucket][index][1] = value
    def get(self, key):
        bucket, index = self.key_index(key)
        return -1 if index == -1 else self.hashmap[bucket][index][1]
    def remove(self, key):
        bucket, index = self.key_index(key)
        if index != -1:
            del self.hashmap[bucket][index]
    def hash_function(self, key):
        return key % self.size
    def key_index(self, key):           
        bucket = self.hash_function(key)
        pairs = self.hashmap[bucket]
        for i in range(len(pairs)):
            if pairs[i][0] == key:
                return (bucket, i)
        return (bucket, -1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isSelfCrossing(self, x):
        for i in range(len(x)):
            if i >= 3:
                if x[i - 1] <= x[i - 3] and x[i] >= x[i - 2]:
                    return True
            if i >= 4:
                if x[i - 1] == x[i - 3] and x[i] >= x[i - 2] - x[i - 4]:
                    return True
            if i >= 5:
                if x[i] >= x[i - 2] - x[i - 4] and x[i - 2] > x[i - 4] and x[i - 3] - x[i - 5] <= x[i - 1] <= x[i - 3]:
                    return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minDeletionSize(self, A):
        rows, cols = len(A), len(A[0])
        cols_deleted = 0
        rows_to_check = {row for row in range(1, rows)}     
        for col in range(cols):
            new_checks = set(rows_to_check)
            for row in rows_to_check:
                char = A[row][col]
                prev_char = A[row - 1][col]
                if char < prev_char:                        
                    cols_deleted += 1
                    break
                elif char > prev_char:                      
                    new_checks.remove(row)
            else:                                           
                if not new_checks:                          
                    break
                rows_to_check = new_checks                  
        return cols_deleted
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        for r in range(rows - 1):
            for c in range(cols - 1):
                if matrix[r][c] != matrix[r + 1][c + 1]:
                    return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isValid(self, code):
        status = "text"
        tag_stack = []
        upper = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   
        i = 0
        while i < len(code):
            c = code[i]
            if status == "text":
                if c == "<":
                    if i + 1 < len(code) and code[i + 1] == "/":
                        status = "closing"
                        i += 2
                        tag_start = i       
                    elif i + 8 < len(code) and code[i + 1:i + 9] == "![CDATA[" and tag_stack:   
                        status = "cdata"
                        i += 9
                    else:
                        status = "opening"
                        i += 1
                        tag_start = i
                elif not tag_stack:         
                    return False
                else:
                    i += 1
            elif status in ["opening", "closing"]:
                if code[i] == ">":
                    tag = code[tag_start:i]
                    if len(tag) < 1 or len(tag) > 9:
                        return False
                    if status == "opening":
                        tag_stack.append(tag)
                    else:
                        if not tag_stack or tag_stack.pop() != tag:
                            return False
                        if not tag_stack and i != len(code) - 1:    
                            return False
                    status = "text"
                elif c not in upper:
                    return False
                i += 1
            elif status == "cdata":
                if i + 2 < len(code) and code[i:i + 3] == "]]>":
                    i += 3
                    status = "text"
                else:
                    i += 1
        return status == "text" and not tag_stack
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findKthLargest(self, nums, k):
        k = len(nums) - k               
        left, right = 0, len(nums)-1
        while True:
            index = self.partition(nums, left, right)   
            if index == k:
                return nums[index]
            if index > k:
                right = index-1
            else:
                left = index+1
    def partition(self, nums, left, right):
        rand_index = random.randint(left, right)
        rand_entry = nums[rand_index]
        nums[rand_index], nums[right] = nums[right], nums[rand_index]   
        next_lower = left       
        for i in range(left, right):
            if nums[i] <= rand_entry:   
                nums[next_lower], nums[i] = nums[i], nums[next_lower]
                next_lower += 1
        nums[next_lower], nums[right] = nums[right], nums[next_lower]   
        return next_lower
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def smallestDistancePair(self, nums, k):
        def k_pair_distances(diff):
            count, j = 0, 0
            for i, num in enumerate(nums):      
                while num - nums[j] > diff:     
                    j += 1
                count += i - j                  
            return count >= k
        nums.sort()
        left, right = 0, nums[-1] - nums[0]
        while left < right:
            mid = (left + right) // 2
            if k_pair_distances(mid):   
                right = mid
            else:
                left = mid + 1          
        return left
EOF
def solve(meal_cost, tip_percent, tax_percent):
    total_cost = round(meal_cost + (meal_cost * (tip_percent/100)) + (meal_cost * (tax_percent/100)))
    print(total_cost)
if __name__ == '__main__':
    meal_cost = float(input())
    tip_percent = int(input())
    tax_percent = int(input())
    solve(meal_cost, tip_percent, tax_percent)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyCircularQueue(object):
    def __init__(self, k):
        self.k = k + 1              
        self.q = [None] * self.k
        self.head = self.tail = 0   
    def enQueue(self, value):
        if self.isFull():
            return False
        self.q[self.tail] = value
        self.tail = (self.tail - 1) % self.k
        return True
    def deQueue(self):
        if self.isEmpty():
            return False
        self.head = (self.head - 1) % self.k  
        return True
    def Front(self):
        if self.isEmpty():
            return -1
        return self.q[self.head]
    def Rear(self):
        if self.isEmpty():
            return -1
        return self.q[(self.tail + 1) % self.k]
    def isEmpty(self):
        return self.head == self.tail
    def isFull(self):
        return (self.head + 1) % self.k == self.tail
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TwoSum(object):
    def __init__(self):
        self.nums = {}      
    def add(self, number):
        self.nums[number] = number in self.nums     
    def find(self, value):
        for num in self.nums:
            if value == 2 * num:        
                if self.nums[num]:
                    return True
            else:                       
                if value - num in self.nums:
                    return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def palindromePairs(self, words):
        palindromes = []
        word_to_index = {}
        for i, word in enumerate(words):
            word_to_index[word] = i
        for i, word in enumerate(words):
            for first_right in range(len(word) + 1):
                left, right = word[:first_right], word[first_right:]
                rev_left, rev_right = left[::-1], right[::-1]
                if first_right != 0 and left == rev_left and rev_right in word_to_index and word_to_index[rev_right] != i:
                    palindromes.append([word_to_index[rev_right], i])
                if right == rev_right and rev_left in word_to_index and word_to_index[rev_left] != i:
                    palindromes.append([i, word_to_index[rev_left]])
        return palindromes
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def judgeCircle(self, moves):
        return moves.count("U") == moves.count("D") and moves.count("L") == moves.count("R")
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def compress(self, chars):
        chars += " "            
        char_start = 0          
        result_length = 0       
        for i, c in enumerate(chars):
            if c != chars[char_start]:      
                chars[result_length] = chars[char_start]    
                result_length += 1
                seq_length = i - char_start
                if seq_length > 1:          
                    digits = list(str(seq_length))
                    digits_length = len(digits)
                    chars[result_length:result_length + digits_length] = digits 
                    result_length += digits_length
                char_start = i
        return result_length
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minMoves(self, nums):
        return sum(nums) - min(nums) * len(nums)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMinDifference(self, timePoints):
        minutes = []
        for time in timePoints:
            hrs, mins = time.split(":")
            minutes.append(int(hrs) * 60 + int(mins))
        minutes.sort()
        minutes.append(minutes[0] + 24 * 60)  
        min_diff = 24 * 60
        for i in range(1, len(minutes)):
            min_diff = min(min_diff, minutes[i] - minutes[i - 1])
        return min_diff
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def groupAnagrams(self, strs):
        sorted_words = defaultdict(list)
        for word in strs:
            letter_list = [c for c in word]
            letter_list.sort()
            sorted_word = "".join(letter_list)
            sorted_words[sorted_word].append(word)
        return list(sorted_words.values())
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minAreaFreeRect(self, points):
        min_area = float("inf")
        points = [complex(*p) for p in sorted(points)]
        line_to_mid = defaultdict(list)
        for p1, p2 in combinations(points, 2):
            line_to_mid[p2 - p1].append((p1 + p2) / 2)
        for line1, mid_points in line_to_mid.items():
            for mid1, mid2 in combinations(mid_points, 2):
                line2 = mid2 - mid1
                if line1.real * line2.real + line1.imag * line2.imag == 0:
                    min_area = min(min_area, abs(line1) * abs(line2))
        return min_area if min_area != float("inf") else 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def boundaryOfBinaryTree(self, root):
        def left_side(node):
            if not node or (not node.left and not node.right):
                return
            boundary.append(node.val)
            if node.left:
                left_side(node.left)
            else:
                left_side(node.right)
        def right_side(node):
            if not node or (not node.left and not node.right):
                return
            right_edge.append(node.val)
            if node.right:
                right_side(node.right)
            else:
                right_side(node.left)
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            if not node.left and not node.right:
                boundary.append(node.val)
            inorder(node.right)
        if not root:
            return []
        boundary, right_edge = [root.val], []
        left_side(root.left)
        inorder(root.left)
        inorder(root.right)
        right_side(root.right)
        return boundary + right_edge[::-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class HitCounter(object):
    def __init__(self):
        self.time_diff = 300
        self.q = deque()
    def hit(self, timestamp):
        self.q.append(timestamp)
    def getHits(self, timestamp):
        while self.q and timestamp - self.q[0] >= self.time_diff:
            self.q.popleft()
        return len(self.q)
class HitCounter2(object):
    def __init__(self):
        self.time_diff = 300
        self.q = deque()
        self.count = 0
    def hit(self, timestamp):
        if self.q and self.q[len(self.q) - 1][0] == timestamp:
            self.q[len(self.q) - 1][1] += 1
        else:
            self.q.append([timestamp, 1])
        self.count += 1
    def getHits(self, timestamp):
        while self.q and timestamp - self.q[0][0] >= self.time_diff:
            _, num = self.q.popleft()
            self.count -= num
        return self.count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class SnakeGame(object):
    def __init__(self, width,height,food):
        self.rows, self.cols = height, width
        self.food = [tuple(f) for f in food]    
        self.snake = {(0, 0) : None}            
        self.head, self.tail = (0, 0), (0, 0)
        self.moves = {"U" : (-1, 0), "D" : (1, 0), "L" : (0, -1), "R" : (0, 1)}
    def move(self, direction):
        new_head = (self.head[0] + self.moves[direction][0], self.head[1] + self.moves[direction][1])
        if new_head[0] < 0 or new_head[0] >= self.rows or new_head[1] < 0 or new_head[1] >= self.cols:
            return -1
        if new_head in self.snake and new_head != self.tail:
            return -1
        self.snake[self.head] = new_head
        self.head = new_head
        if len(self.snake) - 1 >= len(self.food) or new_head != self.food[len(self.snake) - 1]:
            old_tail = self.tail
            self.tail = self.snake[self.tail]
            del self.snake[old_tail]    
        self.snake[self.head] = None
        return len(self.snake) - 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Codec:
    def serialize(self, root):
        nodes = []
        def preorder(node):
            if not node:
                nodes.append("null")
            else:
                nodes.append(str(node.val))
                preorder(node.left)
                preorder(node.right)
        preorder(root)
        return ",".join(nodes)  
    def deserialize(self, data):
        node_list = deque(data.split(","))
        def rebuild():
            if not node_list:
                return None
            node = node_list.popleft()
            if node == "null":
                return None
            node = TreeNode(node)
            node.left = rebuild()
            node.right = rebuild()
            return node
        return rebuild()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProfit(self, prices):
        buy = float('-inf')     
        sell = 0                
        for price in prices:
            buy = max(-price, buy)          
            sell = max(price + buy, sell)   
        return sell
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def romanToInt(self, s):
        doubles = {'CM' : 900, 'CD' : 400, 'XC' : 90, 'XL' : 40, 'IX' :9, 'IV' : 4}
        singles = {'M' : 1000, 'D' : 500, 'C' : 100, 'L' : 50, 'X' : 10, 'V' : 5, 'I' : 1}
        integer = 0
        i = 0
        while i < len(s):
            if i < len(s) - 1 and s[i:i + 2] in doubles:
                integer += doubles[s[i:i + 2]]
                i += 2
            else:
                integer += singles[s[i]]
                i += 1
        return integer
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def searchRange(self, nums, target):
        def binary(target, left, right):
            if left > right:
                return left
            mid = (left + right) // 2
            if target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
            return binary(target, left, right)
        lower = binary(target - 0.5, 0, len(nums) - 1)
        upper = binary(target + 0.5, 0, len(nums) - 1)
        return [-1, -1] if lower == upper else [lower, upper - 1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def rangeSumBST(self, root, L, R):
        def helper(node):
            if not node:
                return 0
            if node.val > R:
                return helper(node.left)
            if node.val < L:
                return helper(node.right)
            return node.val + helper(node.left) + helper(node.right)
        return helper(root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def addDigits(self, num):
        while num > 9:
            num = sum([int(c) for c in str(num)])
        return num
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxProfit(self, prices):
        buy, sell, prev_sell = float("-inf"), 0, 0
        for i, price in enumerate(prices):
            buy = max(buy, prev_sell-price)
            prev_sell = sell
            sell = max(sell, buy+price)
            print(buy, sell, prev_sell)
        return sell
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def deleteDuplicates(self, head):
        pseudo = prev = ListNode(None)
        pseudo.next = head
        node = head
        while node:
            if node.next and node.val == node.next.val:     
                duplicate_value = node.val
                node = node.next
                while node and node.val == duplicate_value: 
                    node = node.next
                prev.next = None                            
            else:                   
                prev.next = node    
                prev = node
                node = node.next
        return pseudo.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def postorderTraversal(self, root):
        if not root:
            return []
        result = deque()        
        stack = [root]
        while stack:
            node = stack.pop()
            result.appendleft(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return list(result)
class Solution2(object):
    def postorderTraversal(self, root):
        result = []
        self.postorder(root, result)
        return result
    def postorder(self, node, result):
        if not node:
            return
        self.postorder(node.left, result)
        self.postorder(node.right, result)
        result.append(node.val)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def upsideDownBinaryTree(self, root):
        if not root or not root.left:       
            return root
        new_root = self.upsideDownBinaryTree(root.left)     
        node = new_root
        while node.right:       
            node = node.right
        node.left = root.right
        node.right = root
        root.left = None
        root.right = None
        return new_root
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def kInversePairs(self, n, k):
        if k == 0:
            return 1
        MODULO = 10 ** 9 + 7
        inverse_pairs = [0 for _ in range(k + 1)]
        for num in range(1, n + 1):  
            next_inverse_pairs = [1]  
            for nb_pairs in range(1, k + 1):
                next_inverse_pairs.append(next_inverse_pairs[-1])  
                next_inverse_pairs[-1] += inverse_pairs[nb_pairs]  
                if nb_pairs - num >= 0:  
                    next_inverse_pairs[-1] -= inverse_pairs[nb_pairs - num]
                next_inverse_pairs[-1] %= MODULO
            inverse_pairs = next_inverse_pairs
        return (inverse_pairs[-1] - inverse_pairs[-2]) % MODULO  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def integerBreak(self, n):
        if n <= 3:
            return n - 1
        threes, remainder = divmod(n - 4, 3)
        product = 3**threes
        remainder += 4
        if remainder == 4:
            return product * 2 * 2
        if remainder == 5:
            return product * 3 * 2
        return product * 3 * 3
class Solution2(object):
    def integerBreak(self, n):
        max_breaks = [0, 1]
        for i in range(2, n + 1):
            max_break = 0
            for j in range(1, (i // 2) + 1):
                max_break = max(max_break, max(j, max_breaks[j]) * max(i - j, max_breaks[i - j]))
            max_breaks.append(max_break)
        return max_breaks[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def superPow(self, a, b):
        result = 1
        for digit in b:         
            result = (pow(result, 10, 1337) * pow(a, digit, 1337)) % 1337
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wordSquares(self, words):
        prefixes = defaultdict(list)
        for word in words:
            for i in range(1, len(word)):           
                prefixes[word[:i]].append(word)
        squares = []
        for word in words:
            self.build_square([word], prefixes, squares)    
        return squares
    def build_square(self, partial, prefixes, squares):
        if len(partial) == len(partial[0]):                 
            squares.append(list(partial))                   
            return
        prefix = []
        col = len(partial)
        for row in range(len(partial)):
            prefix.append(partial[row][col])
        next_words = prefixes["".join(prefix)]
        for next_word in next_words:
            partial.append(next_word)
            self.build_square(partial, prefixes, squares)
            partial.pop()       
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def brokenCalc(self, X, Y):
        operations = 0
        while Y > X:
            operations += 1
            if Y % 2 == 0:
                Y //= 2
            else:
                Y += 1
        return X - Y + operations
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wordsTyping(self, sentence, rows, cols):
        sentence_len = sum(len(w) for w in sentence) + len(sentence)
        line_fits = []
        for start_word_index in range(len(sentence)):
            row_length, sentences = 0, 0
            word_index = start_word_index
            while row_length + sentence_len <= cols:  
                row_length += sentence_len
                sentences += 1
            while row_length + len(sentence[word_index]) <= cols:  
                row_length += len(sentence[word_index]) + 1  
                word_index += 1  
                if word_index == len(sentence):  
                    sentences += 1
                    word_index = 0
            line_fits.append((sentences, word_index))
        fits, word_index = 0, 0
        for r in range(rows):
            sentences, next_word_index = line_fits[word_index]
            fits += sentences
            word_index = next_word_index
        return fits
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def superpalindromesInRange(self, L, R):
        L_sqrt = int(int(L) ** 0.5)
        R_sqrt = int((int(R) + 1) ** 0.5)
        digits = [str(i) for i in range(10)]
        def is_palindrome(i):
            return str(i) == str(i)[::-1]
        prev_palis, palis = [""], digits[:]         
        result = sum(L_sqrt <= i <= R_sqrt and is_palindrome(i ** 2) for i in range(10))
        for _ in range(2, 11):                      
            new_palis = []
            for digit in digits:
                for pal in prev_palis:              
                    new_pal = digit + pal + digit   
                    new_palis.append(new_pal)
                    if new_pal[0] == "0":           
                        continue
                    num = int(new_pal)
                    if num > R_sqrt:                
                        return result
                    if L_sqrt <= num and is_palindrome(num ** 2):   
                        result += 1
            prev_palis, palis = palis, new_palis
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def constructArray(self, n, k):
        result = []
        low, high = 1, k + 1
        next_low = True
        while low <= high:
            if next_low:
                result.append(low)
                low += 1
            else:
                result.append(high)
                high -= 1
            next_low = not next_low
        return result + list(range(k + 2, n + 1))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findUnsortedSubarray(self, nums):
        n = len(nums)
        right, left = -1, -1        
        for i in range(1, n):
            if left == - 1 and nums[i] < nums[i - 1]:
                left = i - 1        
                min_num = nums[i]
            elif left != -1:        
                min_num = min(min_num, nums[i])
        if left == -1:              
            return 0
        for i in range(n - 2, -1, -1):
            if right == -1 and nums[i] > nums[i + 1]:
                right = i + 1       
                max_num = nums[i]
            elif right != -1:       
                max_num = max(max_num, nums[i])
        while left > 0 and nums[left - 1] > min_num:        
            left -= 1
        while right < n - 1 and nums[right + 1] < max_num:  
            right += 1
        return right - left + 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestIncreasingPath(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        longest = 0
        memo = [[-1 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                longest = max(longest, self.dfs(r, c, matrix, memo))
        return longest
    def dfs(self, r, c, matrix, memo):
        if memo[r][c] != -1:
            return memo[r][c]
        longest_here = 1        
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= r + dr < len(matrix) and 0 <= c + dc < len(matrix[0]) and matrix[r + dr][c + dc] > matrix[r][c]:
                longest_here = max(longest_here, self.dfs(r + dr, c + dc, matrix, memo) + 1)
        memo[r][c] = longest_here
        return longest_here
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def uniqueMorseRepresentations(self, words):
        codes = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.",
                 "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        morse = set()
        for word in words:
            transformation = []
            for c in word:
                transformation.append(codes[ord(c) - ord("a")])
            morse.add("".join(transformation))
        return len(morse)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def crackSafe(self, n, k):
        seen = set()
        digits = [str(i) for i in range(k)]
        result = []
        def dfs(node):
            for x in digits:
                pattern = node + x
                if pattern not in seen:
                    seen.add(pattern)
                    dfs(pattern[1:])
                    result.append(x)
        dfs("0" * (n - 1))
        return "".join(result) + "0" * (n - 1)
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def deleteNode(head, position):
    if position == 0:
        return head.next
    cur = head
    for _ in range(position-1):
        cur = cur.next
    cur.next = cur.next.next
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)
    position = int(input())
    llist1 = deleteNode(llist.head, position)
    print_singly_linked_list(llist1, ' ', fptr)
    fptr.write('\n')
    fptr.close()
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def maxTurbulenceSize(self, A):
        if len(A) == 1:             
            return 1
        result = 0
        down, up = 0, 0             
        for i in range(1, len(A)):
            if A[i] > A[i - 1]:     
                up += 1
                down = 0
            elif A[i] < A[i - 1]:   
                down += 1
                up = 0
            else:                   
                down = 0
                up = 0
            result = max(result, up, down)
            up, down = down, up     
        return result + 1 if result != 0 else 0     
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkRecord(self, s):
        return s.count("A") < 2 and "LLL" not in s
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def jump(self, nums):
        if len(nums) == 1:
            return 0
        start, end = 0, 0   
        max_index = 0
        steps = 1
        while True:         
            for i in range(start, end+1):
                max_index = max(max_index, i + nums[i])
            if max_index >= len(nums)-1:
                return steps
            steps += 1
            start, end = end + 1, max_index
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def longestMountain(self, A):
        valley, peak = 0, 0
        prev = 0                    
        longest = 0
        for i in range(1, len(A)):
            if A[i] == A[i - 1]:    
                valley, peak = i, i
                prev = 0
            elif A[i] > A[i - 1]:
                if prev == 1:       
                    peak = i
                else:               
                    valley = i - 1
                prev = 1
            elif A[i] < A[i - 1]:
                if prev == 1:       
                    peak = i - 1
                    longest = max(longest, i - valley + 1)
                else:               
                    if peak > valley:
                        longest = max(longest, i - valley + 1)
                prev = -1
        return longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rotate(self, nums, k):
        k %= len(nums)
        nums.reverse()
        nums[:k] = reversed(nums[:k])
        nums[k:] = reversed(nums[k:])
class Solution2(object):
    def rotate(self, nums, k):
        n = len(nums)
        k %= n
        nums[:] = nums[n-k:] + nums[:n-k]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def calculateMinimumHP(self, dungeon):
        rows, cols = len(dungeon), len(dungeon[0])
        for r in range(rows - 1):       
            dungeon[r].append(float('inf'))
        dungeon.append([float('inf') for _ in range(cols + 1)])
        dungeon[rows - 1].append(1)     
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                dungeon[r][c] = max(1, -dungeon[r][c] + min(dungeon[r+1][c], dungeon[r][c+1]))
        return dungeon[0][0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        max_consecutive = 0
        i = 0
        while i < len(nums) and nums[i] == 0:   
            i += 1
        start, prev_start = i, max(i - 1, 0)
        for j in range(i + 1, len(nums)):
            if nums[j] == 0:
                if j != 0 and nums[j - 1] == 0:
                    prev_start = j      
                else:
                    max_consecutive = max(max_consecutive, j - prev_start)
                    prev_start = start  
                start = j + 1           
        return max(max_consecutive, len(nums) - prev_start)     
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def cleanRoom(self, robot):
        visited = set()
        def dfs(x, y, dx, dy):
            robot.clean()
            visited.add((x, y))
            for _ in range(4):
                if ((x + dx, y + dy)) not in visited and robot.move():
                    dfs(x + dx, y + dy, dx, dy)
                    robot.turnLeft()        
                    robot.turnLeft()
                    robot.move()
                    robot.turnLeft()
                    robot.turnLeft()
                robot.turnLeft()
                dx, dy = -dy, dx
        dfs(0, 0, 0, 1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MapSum(object):
    def __init__(self):
        self.dict = defaultdict(int)        
        self.words = defaultdict(int)       
    def insert(self, key, val):
        if key in self.words:               
            self.words[key], val = val, val - self.words[key]
        else:
            self.words[key] = val           
        for i in range(len(key)):
            prefix = key[:i + 1]
            self.dict[prefix] += val        
    def sum(self, prefix):
        return self.dict[prefix]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLUSlength(self, strs):
        def is_subsequence(s, t):       
            i, j = 0, 0
            while i < len(s) and j < len(t):
                if s[i] == t[j]:
                    i += 1
                j += 1
            if i == len(s):
                return True
            return False
        counts = Counter(strs)
        unique_strs = list(counts.keys())
        unique_strs.sort(key=len, reverse=True)
        seen = set()
        for s in unique_strs:
            if counts[s] == 1:
                if not any([is_subsequence(s, t) for t in seen]):
                    return len(s)
            else:
                seen.add(s)
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        start, longest = 0, 0
        last_seen = defaultdict(int)        
        for end, c in enumerate(s):
            last_seen[c] = end                      
            while len(last_seen) > k:               
                if last_seen[s[start]] == start:    
                    del last_seen[s[start]]         
                start += 1                          
            else:                           
                longest = max(longest, end - start + 1)
        return longest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findRotateSteps(self, ring, key):
        def dist(i, j):
            return min(abs(i - j), len(ring) - abs(i - j))
        char_to_ring = defaultdict(list)    
        for i, c in enumerate(ring):
            char_to_ring[c].append(i)
        i_to_steps = {0: 0}  
        for k in key:
            new_i_to_steps = {}
            new_indices = char_to_ring[k]  
            for new_i in new_indices:
                min_steps = float("inf")
                for i in i_to_steps:
                    min_steps = min(min_steps, i_to_steps[i] + dist(i, new_i))
                new_i_to_steps[new_i] = min_steps
            i_to_steps = new_i_to_steps
        return min(i_to_steps.values()) + len(key)      
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minDistance(self, height, width, tree, squirrel, nuts):
        if not nuts:
            return 0
        nuts_to_tree = 0  
        best_gain = height * width  
        def distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        for nut in nuts:
            nut_to_tree = distance(nut, tree)
            squirrel_to_nut = distance(squirrel, nut)
            nuts_to_tree += nut_to_tree
            best_gain = min(best_gain, squirrel_to_nut - nut_to_tree)
        return 2 * nuts_to_tree + best_gain
EOF
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        return all(i == 0 or j == 0 or matrix[i-1][j-1] == val
                   for i, row in enumerate(matrix)
                   for j, val in enumerate(row))
class Solution2(object):
    def isToeplitzMatrix(self, matrix):
        for row_index, row in enumerate(matrix):
            for digit_index, digit in enumerate(row):
                if not row_index or not digit_index:
                    continue
                if matrix[row_index - 1][digit_index - 1] != digit:
                    return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minCost(self, costs):
        if not costs:
            return 0
        for i in range(1, len(costs)):
            costs[i][0] += min(costs[i-1][1], costs[i-1][2])
            costs[i][1] += min(costs[i-1][0], costs[i-1][2])
            costs[i][2] += min(costs[i-1][0], costs[i-1][1])
        return min(costs[-1])
EOF
class Solution(object):
    def fractionAddition(self, expression):
        def gcd(a, b):
            while b:
                a, b = b, a%b
            return a
        ints = map(int, re.findall('[+-]?\d+', expression))
        A, B = 0, 1
        for i in xrange(0, len(ints), 2):
            a, b = ints[i], ints[i+1]
            A = A * b + a * B
            B *= b
            g = gcd(A, B)
            A //= g
            B //= g
        return '%d/%d' % (A, B)
EOF
class Solution(object):
    def findCelebrity(self, n):
        candidate = 0
        for i in xrange(1, n):
            if knows(candidate, i):  
                candidate = i        
        for i in xrange(n):
            candidate_knows_i = knows(candidate, i) 
            i_knows_candidate = knows(i, candidate) 
            if i != candidate and (candidate_knows_i or
                                   not i_knows_candidate):
                return -1
        return candidate
EOF
class Solution(object):
    def stoneGame(self, piles):
        if len(piles) % 2 == 0 or len(piles) == 1:
            return True
        dp = [0] * len(piles)
        for i in reversed(xrange(len(piles))):
            dp[i] = piles[i]
            for j in xrange(i+1, len(piles)):
                dp[j] = max(piles[i] - dp[j], piles[j] - dp[j - 1])
        return dp[-1] >= 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def updateMatrix(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        frontier = deque()  
        max_dist = max(rows, cols)
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 1:
                    matrix[r][c] = max_dist
                else:
                    frontier.append((r, c))
        while frontier:
            r, c = frontier.popleft()
            for dr, dc in deltas:
                if 0 <= r + dr < rows and 0 <= c + dc < cols and matrix[r][c] + 1 < matrix[r + dr][c + dc]:
                    matrix[r + dr][c + dc] = matrix[r][c] + 1
                    frontier.append((r + dr, c + dc))
        return matrix
class Solution2(object):
    def updateMatrix(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        unknown = set()  
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 1:
                    unknown.add((r, c))
        while unknown:
            new_unknown = set()
            for r, c in unknown:
                for dr, dc in deltas:
                    if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in unknown:
                        matrix[r][c] = matrix[r + dr][c + dc] + 1
                        break
                else:  
                    new_unknown.add((r, c))
            unknown = new_unknown
        return matrix
EOF
vowels = ['A', 'E', 'I', 'O', 'U']
def minion_game(string):
    score_kevin = 0
    score_stuart = 0
    for ind in range(len(string)):
        if string[ind] in vowels:
            score_kevin += len(string) - ind
        else:
            score_stuart += len(string) - ind
    if score_kevin > score_stuart:
        print("Kevin {}".format(score_kevin))
    elif score_kevin < score_stuart:
        print("Stuart {}".format(score_stuart))
    else:
        print("Draw")
EOF
if __name__ == "__main__":
    N = int(input().strip())
    stamps = set()
    for _ in range(N):
        stamp = input().strip()
        stamps.add(stamp)
    print(len(stamps))
EOF
def solve(a0, a1, a2, b0, b1, b2):
    alice = bob = 0
    if a0 > b0:
        alice += 1
    elif a0 < b0:
        bob += 1
    if a1 > b1:
        alice += 1
    elif a1 < b1:
        bob += 1
    if a2 > b2:
        alice += 1
    elif a2 < b2:
        bob += 1
    return alice, bob
a0, a1, a2 = input().strip().split(' ')
a0, a1, a2 = [int(a0), int(a1), int(a2)]
b0, b1, b2 = input().strip().split(' ')
b0, b1, b2 = [int(b0), int(b1), int(b2)]
result = solve(a0, a1, a2, b0, b1, b2)
print (" ".join(map(str, result)))
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def insertNodeAtPosition(head, data, position):
    Node = SinglyLinkedListNode(data)
    if position == 0:
        Node.next = head
        head = Node
        return head
    prev = None
    cur = head
    for i in range(position):
        prev = cur
        cur = cur.next
    Node.next = prev.next
    prev.next = Node
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)
    data = int(input())
    position = int(input())
    llist_head = insertNodeAtPosition(llist.head, data, position)
    print_singly_linked_list(llist_head, ' ', fptr)
    fptr.write('\n')
    fptr.close()
EOF
class CombinationIterator(object):
    def __init__(self, characters, combinationLength):
        self.__it = itertools.combinations(characters, combinationLength)
        self.__curr = None
        self.__last = characters[-combinationLength:]
    def next(self):
        self.__curr = "".join(self.__it.next())
        return self.__curr
    def hasNext(self):
        return self.__curr != self.__last
class CombinationIterator2(object):
    def __init__(self, characters, combinationLength):
        self.__characters = characters
        self.__combinationLength = combinationLength
        self.__it = self.__iterative_backtracking()
        self.__curr = None
        self.__last = characters[-combinationLength:]
    def __iterative_backtracking(self):
        def conquer():
            if len(curr) == self.__combinationLength:
                return curr
        def prev_divide(c):
            curr.append(c)
        def divide(i):
            if len(curr) != self.__combinationLength:
                for j in reversed(xrange(i, len(self.__characters)-(self.__combinationLength-len(curr)-1))):
                    stk.append(functools.partial(post_divide))
                    stk.append(functools.partial(divide, j+1))
                    stk.append(functools.partial(prev_divide, self.__characters[j]))
            stk.append(functools.partial(conquer))
        def post_divide():
            curr.pop()
        curr = []
        stk = [functools.partial(divide, 0)]
        while stk:
            result = stk.pop()()
            if result is not None:
                yield result
    def next(self):
        self.__curr = "".join(next(self.__it))
        return self.__curr
    def hasNext(self):
        return self.__curr != self.__last
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def mincostTickets(self, days: 'List[int]', costs: 'List[int]') -> 'int':
        PASSES = [1, 7, 30]         
                def get_min_cost(i):        
            if i >= len(days):
                return 0
            min_cost = float("inf")
            j = i
            for length, cost in zip(PASSES, costs):
                while j < len(days) and days[j] < days[i] + length:     
                    j += 1
                min_cost = min(min_cost, cost + get_min_cost(j))
            return min_cost
        return get_min_cost(0)
EOF
class Solution(object):
    def jump(self, A):
        jump_count = 0
        reachable = 0
        curr_reachable = 0
        for i, length in enumerate(A):
            if i > reachable:
                return -1
            if i > curr_reachable:
                curr_reachable = reachable
                jump_count += 1
            reachable = max(reachable, i + length)
        return jump_count
EOF
class Solution(object):
    def missingNumber(self, arr):
        def check(arr, d, x):
            return arr[x] != arr[0] + d*x
        d = (arr[-1]-arr[0])//len(arr)
        left, right = 0, len(arr)-1
        while left <= right:
            mid = left + (right-left)//2
            if check(arr, d, mid):
                right = mid-1
            else:
                left = mid+1
        return arr[0] + d*left
class Solution2(object):
    def missingNumber(self, arr):
        return (min(arr)+max(arr))*(len(arr)+1)//2 - sum(arr)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countPalindromicSubsequences(self, S):
        NUM_LETTERS, MOD = 4, 10 ** 9 + 7
        S = [ord(c) - ord("a") for c in S]                  
        memo = {}
        last_indices = [-1 for _ in range(NUM_LETTERS)]
        prev_index_letter = [None for _ in
                             range(len(S))]                 
        for i in range(len(S)):                             
            last_indices[S[i]] = i
            prev_index_letter[i] = last_indices[:]
        last_indices = [-1 for _ in range(NUM_LETTERS)]
        next_index_letter = [None for _ in range(len(S))]   
        for i in range(len(S) - 1, -1, -1):                 
            last_indices[S[i]] = i
            next_index_letter[i] = last_indices[:]
        def helper(i, j):                                   
            if (i, j) in memo:
                return memo[(i, j)]
            count = 1                                       
            for letter in range(4):
                next_index = next_index_letter[i][letter]   
                prev_index = prev_index_letter[j][letter]   
                if i <= next_index <= j:                    
                    count += 1
                if next_index != -1 and prev_index != -1 and prev_index > next_index:   
                    count += helper(next_index + 1, prev_index - 1)
            count %= MOD
            memo[(i, j)] = count
            return count
        return helper(0, len(S) - 1) - 1                    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findDuplicate(self, paths):
        content_to_path = defaultdict(list)
        for path in paths:
            path_list = path.split(" ")     
            for f in path_list[1:]:
                open_bracket = f.index("(")
                close_bracket = f.index(")")
                content = f[open_bracket + 1:close_bracket]
                content_to_path[content].append(path_list[0] + "/" + f[:open_bracket])
        return [dup for dup in content_to_path.values() if len(dup) > 1]    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largeGroupPositions(self, S):
        result = []
        start = 0
        for i, c in enumerate(S):
            if i == len(S) - 1 or c != S[i + 1]:
                if i - start >= 2:
                    result.append([start, i])
                start = i + 1                       
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MyCalendar(object):
    def __init__(self):
        self.bookings = [(float("-inf"), float("-inf")), (float("inf"), float("inf"))]
    def book(self, start, end):
        i = bisect.bisect_left(self.bookings, (start, end))
        if end > self.bookings[i][0]:
            return False
        if start < self.bookings[i - 1][1]:
            return False
        self.bookings.insert(i, (start, end))
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def knightProbability(self, N, K, r, c):
        M = N // 2              
        if N % 2 == 1:          
            M += 1
        def convert(r1, c1):    
            if r1 >= M:
                r1 = N - 1 - r1
            if c1 >= M:
                c1 = N - 1 - c1
            return [r1, c1]
        probs = [[1 for _ in range(M)] for _ in range(M)]       
        for _ in range(K):
            new_probs = [[0 for _ in range(M)] for _ in range(M)]
            for r1 in range(M):
                for c1 in range(M):
                    prob = 0
                    for dr in [2, 1, -1, -2]:                   
                        for dc in [3 - abs(dr), abs(dr) - 3]:
                            if 0 <= r1 + dr < N and 0 <= c1 + dc < N:   
                                r2, c2 = convert(r1 + dr, c1 + dc)
                                prob += probs[r2][c2] / 8.0             
                    new_probs[r1][c1] = prob                    
            probs = new_probs                                   
        r, c = convert(r, c)
        return probs[r][c]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        nb_prerequisites = defaultdict(int)     
        prereq_list = defaultdict(list)         
        for after, before in prerequisites:
            nb_prerequisites[after] += 1
            prereq_list[before].append(after)
        can_take = set(i for i in range(numCourses)) - set(nb_prerequisites.keys())
        while can_take:
            course = can_take.pop()                     
            numCourses -= 1                             
            for dependent in prereq_list[course]:
                nb_prerequisites[dependent] -= 1        
                if nb_prerequisites[dependent] == 0:    
                    can_take.add(dependent)
        return numCourses == 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numJewelsInStones(self, J, S):
        return sum(1 for s in S if s in set(J))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class IntervalNode(object):
    def __init__(self, interval):
        self.inner = interval
        self.parent = self
class SummaryRanges(object):
    def __init__(self):
        self.parents = {}           
        self.intervals = set()      
    def get_parent(self, v):
        if v not in self.parents:
            return None
        interval_node = self.parents[v]
        while interval_node != interval_node.parent:
            interval_node.parent = interval_node.parent.parent
            interval_node = interval_node.parent
        return interval_node
    def addNum(self, val):
        if val in self.parents:             
            return
        lower = self.get_parent(val - 1)
        upper = self.get_parent(val + 1)
        if lower and upper:                 
            lower.inner.end = upper.inner.end
            self.parents[val] = lower
            upper.parent = lower
            self.intervals.remove(upper.inner)
        elif lower:
            lower.inner.end += 1
            self.parents[val] = lower
        elif upper:
            upper.inner.start -= 1
            self.parents[val] = upper
        else:
            new_inner = Interval(val, val)
            self.parents[val] = IntervalNode(new_inner)
            self.intervals.add(new_inner)
    def getIntervals(self):
        result = list(self.intervals)
        result.sort(key = lambda x : x.start)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isValidSudoku(self, board):
        size = 9
        digits = {str(i) for i in range(1,10)}
        rows = [set() for _ in range(size)]
        cols = [set() for _ in range(size)]
        boxes = [set() for _ in range(size)]
        for r in range(size):
            for c in range(size):
                digit = board[r][c]
                if digit == '.':
                    continue
                if digit not in digits:
                    return False
                box = (size//3) * (r // (size//3)) + (c // (size//3))
                if digit in rows[r] or digit in cols[c] or digit in boxes[box]:
                    return False
                rows[r].add(digit)
                cols[c].add(digit)
                boxes[box].add(digit)
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def threeSumClosest(self, nums, target):
        nums.sort()
        closest = float('inf')  
        for i in range(len(nums) - 2):
            j = i + 1
            k = len(nums) - 1
            while j < k:
                triple = nums[i] + nums[j] + nums[k]
                if triple == target:    
                    return target
                if abs(triple - target) < abs(closest - target):
                    closest = triple
                if triple - target > 0:
                    k -= 1
                else:
                    j += 1
        return closest
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def orangesRotting(self, grid):
        rows, cols = len(grid), len(grid[0])
        fresh, rotten = set(), set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    fresh.add((r, c))
                if grid[r][c] == 2:
                    rotten.add((r, c))
        mins = 0
        while fresh:
            mins += 1
            new_rotten = set()
            for r, c in rotten:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:   
                    if (r + dr, c + dc) in fresh:
                        new_rotten.add((r + dr, c + dc))
            if not new_rotten:
                return -1
            rotten = new_rotten
            fresh -= new_rotten
        return mins
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def checkPerfectNumber(self, num):
        if num <= 1:                                    
            return False
        sum_divisors = 1
        for i in range(2, int(math.sqrt(num)) + 1):     
            div, mod = divmod(num, i)
            if mod == 0:
                sum_divisors += i + div
        return sum_divisors == num
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class KthLargest(object):
    def __init__(self, k, nums):
        heapq.heapify(nums)
        while len(nums) > k:
            heapq.heappop(nums)
        self.k = k
        self.nums = nums
    def add(self, val):
        if len(self.nums) == self.k and val <= self.nums[0]:    
            return self.nums[0]
        heapq.heappush(self.nums, val)
        if len(self.nums) > self.k:
            heapq.heappop(self.nums)
        return self.nums[0]
EOF
class Node:
    def __init__(self,data):
        self.right=self.left=None
        self.data = data
class Solution:
    def insert(self,root,data):
        if root==None:
            return Node(data)
        else:
            if data<=root.data:
                cur=self.insert(root.left,data)
                root.left=cur
            else:
                cur=self.insert(root.right,data)
                root.right=cur
        return root
    def getHeight(self,root):
        if root == None:
            return 0
        elif root.left == None and root.right == None:
            return 0
        return 1 + max(self.getHeight(root.left),self.getHeight(root.right))
T=int(input())
myTree=Solution()
root=None
for i in range(T):
    data=int(input())
    root=myTree.insert(root,data)
height=myTree.getHeight(root)
print(height)       
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxCount(self, m, n, ops):
        max_r, max_c = m, n         
        for r, c in ops:
            max_r = min(max_r, r)
            max_c = min(max_c, c)
        return max_r * max_c
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        savedA, savedB = headA, headB
        while headA != headB:
            headA = savedB if not headA else headA.next
            headB = savedA if not headB else headB.next
        gc.collect()    
        return headA
class Solution2(object):
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        savedA, savedB = headA, headB
        len_diff = 0
        while headA.next:
            len_diff += 1
            headA = headA.next
        while headB.next:
            len_diff -= 1
            headB = headB.next
        if headA != headB:      
            return
        headA, headB = savedA, savedB
        while len_diff != 0: 
            if len_diff > 0:
                headA = headA.next
                len_diff -= 1
            else:
                headB = headB.next
                len_diff += 1
        while headA != headB:
            headA = headA.next
            headB = headB.next
        gc.collect()
        return headA
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def hasCycle(self, head):
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def numSubarraysWithSum(self, A, S):
        result = 0
        running = 0                             
        partials = defaultdict(int, {0: 1})     
        for i, a in enumerate(A):
            running += a
            result += partials[running - S]
            partials[running] += 1              
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxSubArrayLen(self, nums, k):
        cumul, max_length = 0, 0
        first_index = {}
        for i, num in enumerate(nums):
            cumul += num
            if cumul == k:
                max_length = i + 1              
            elif cumul - k in first_index:
                max_length = max(max_length, i - first_index[cumul - k])
            if cumul not in first_index:        
                first_index[cumul] = i
        return max_length
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def validUtf8(self, data):
        i = 0
        while i < len(data):
            byte = data[i]
            if byte & (1 << 7) == 0:        
                i += 1
                continue
            bit = 6
            while byte & (1 << bit) and bit > 3:
                bit -= 1
            if byte & (1 << bit) or bit == 6:   
                return False
            bytes = 6 - bit     
            i += 1
            while bytes:
                if i >= len(data) or data[i] & (128 + 64) != 128:
                    return False
                bytes -= 1
                i += 1
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class MinStack(object):
    def __init__(self):
        self.main = []
        self.mins = []
    def push(self, x):
        self.main.append(x)
        if not self.mins or x <= self.mins[-1]:
            self.mins.append(x)
    def pop(self):
        item = self.main.pop()
        if item == self.mins[-1]:
            self.mins.pop()
    def top(self):
        return self.main[-1]
    def getMin(self):
        return self.mins[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def canWin(self, s):
        def helper(s):
            if s in memo:
                return memo[s]
            for i in range(len(s) - 1):
                if s[i:i + 2] == '++' and not helper(s[:i] + '--' + s[i + 2:]):
                    memo[s] = True
                    return True
            memo[s] = False
            return False
        memo = {}
        return helper(s)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minimumLengthEncoding(self, words):
        required = set(words)
        for word in words:                          
            for i in range(1, len(word) - 1):       
                required.discard(word[i:])          
        return sum(len(w) for w in required) + len(required)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
class Solution(object):
    def outerTrees(self, points):
        if len(points) < 3:
            return points
        def slope(a, b):  
            if a.x == b.x:
                return float("inf")
            return (b.y - a.y) / float(b.x - a.x)
        def cross_product(p):
            v1 = [result[-1].x - result[-2].x, result[-1].y - result[-2].y]
            v2 = [p.x - result[-2].x, p.y - result[-2].y]
            return v1[0] * v2[1] - v1[1] * v2[0]
        start_point = min(points, key=lambda p: (p.x, p.y))
        points.remove(start_point)
        points.sort(key=lambda p: (slope(start_point, p), -p.y, p.x))
        result = [start_point, points[0]]
        for point in points[1:]:
            while cross_product(point) < 0:
                result.pop()
            result.append(point)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def verticalTraversal(self, root):
        x_to_y_and_val = defaultdict(list)
        def helper(node, x, y):
            if not node:
                return
            x_to_y_and_val[x].append((-y, node.val))    
            helper(node.left, x - 1, y - 1)
            helper(node.right, x + 1, y - 1)
        helper(root, 0, 0)
        result = []
        xs = sorted(x_to_y_and_val.keys())
        for x in xs:
            x_to_y_and_val[x].sort()
            result.append([val for _, val in x_to_y_and_val[x]])
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findTarget(self, root, k):
        visited = set()
        def traverse(node):
            if not node:
                return False
            if k - node.val in visited:
                return True
            visited.add(node.val)
            return traverse(node.left) or traverse(node.right)
        return traverse(root)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        rows, cols = len(image), len(image[0])
        startColor = image[sr][sc]
        if startColor == newColor:
            return image
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if image[r][c] != startColor:
                continue
            image[r][c] = newColor
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                stack.append((r + dr, c + dc))
        return image
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def sumNumbers(self, root):
        return self.helper(root, 0)
    def helper(self, node, partial):
        if not node:    
            return 0
        partial = 10 * partial + node.val       
        if not node.left and not node.right:    
            return partial
        return self.helper(node.left, partial) + self.helper(node.right, partial)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def isPowerOfTwo(self, n):
        return n > 0 and not n & (n - 1)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reachNumber(self, target):
        target = abs(target)  
        steps = int(math.ceil((math.sqrt(1 + 8 * target) - 1) / 2)) 
        target -= steps * (steps + 1) // 2                          
        if target % 2 == 0:  
            return steps
        target += steps + 1  
        if target % 2 == 0:
            return steps + 1
        return steps + 2    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if not root or p == root or q == root:      
            return root
        left_lca = self.lowestCommonAncestor(root.left, p, q)
        right_lca = self.lowestCommonAncestor(root.right, p, q)
        if left_lca and right_lca:
            return root
        return left_lca or right_lca
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def partitionLabels(self, S):
        last = {c: i for i, c in enumerate(S)}
        result = []
        start, end = 0, 0           
        for i, c in enumerate(S):
            end = max(end, last[c])
            if i == end:            
                result.append(end - start + 1)
                start = i + 1       
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findMaxForm(self, strs, m, n):
        max_form = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for s in strs:
            s_zeros = sum([True for c in s if c == "0"])
            s_ones = len(s) - s_zeros
            for i in range(m, -1, -1):
                for j in range(n, -1, -1):
                    if i >= s_zeros and j >= s_ones:    
                        max_form[i][j] = max(max_form[i][j], 1 + max_form[i - s_zeros][j - s_ones])
        return max_form[-1][-1]     
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wordBreak(self, s, wordDict):
        can_make = [False] * (len(s)+1)         
        can_make[0] = True
        for i in range(1, len(s)+1):            
            for j in range(i-1, -1, -1):        
                if can_make[j] and s[j:i] in wordDict:
                    can_make[i] = True
                    break
        return can_make[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def simplifyPath(self, path):
        path_list = path.split('/')     
        result = []
        for item in path_list:
            if item == '..':            
                if result:
                    result.pop()
            elif item and item != '.':  
                result.append(item)
        return '/' + '/'.join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxSubarraySumCircular(self, A):
        if all(num <= 0 for num in A):
            return max(A)
        overall_max, overall_min = float('-inf'), float('inf')
        max_ending_here, min_ending_here = 0, 0
        for num in A:
            max_ending_here = max(max_ending_here, 0) + num     
            min_ending_here = min(min_ending_here, 0) + num     
            overall_max = max(overall_max, max_ending_here)
            overall_min = min(overall_min, min_ending_here)
        return max(overall_max, sum(A) - overall_min)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minIncrementForUnique(self, A):
        increments = 0
        last_used = -1
        for num in sorted(A):
            if num <= last_used:
                increments += last_used + 1 - num
                last_used += 1
            else:
                last_used = num
        return increments
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def search(self, reader, target):
        left, right = 0, 20000
        while left <= right:
            mid = (left + right) // 2
            val = reader.get(mid)
            if target == val:
                return mid
            if target > val:
                left = mid + 1
            else:
                right = mid - 1
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def reconstructQueue(self, people):
        queue = []
        height_groups = defaultdict(list)
        for height, in_front in people:
            height_groups[height].append(in_front)
        all_heights = list(height_groups.keys())
        all_heights.sort(reverse = True)
        for height in all_heights:
            height_groups[height].sort()
            for in_front in height_groups[height]:
                queue.insert(in_front, [height, in_front])
        return queue
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numTilings(self, N):
        MOD = 10 ** 9 + 7
        prev_tilings, tilings = 0, 1        
        prev_one_extra, one_extra = 0, 0    
        for _ in range(N):
            next_tilings = (tilings + prev_tilings + prev_one_extra) % MOD
            next_one_extra = (2 * tilings + one_extra) % MOD
            tilings, prev_tilings = next_tilings, tilings
            one_extra, prev_one_extra = next_one_extra, one_extra
        return tilings
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def middleNode(self, head):
        slow, fast = head, head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        return slow
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def convertToTitle(self, n):
        column = deque()    
        while n > 0:
            n, output = divmod(n-1, 26)
            column.appendleft(output)
        return "".join([chr(i+ord('A')) for i in column])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minTransfers(self, transactions):
        balances = defaultdict(int)                 
        for lender, receiver, amount in transactions:
            balances[lender] += amount
            balances[receiver] -= amount
        net_balances = [b for b in balances.values() if b != 0]
        def transfers(net_balances):
            if not net_balances:                    
                return 0
            b = net_balances[0]
            for i in range(1, len(net_balances)):   
                if b == -net_balances[i]:
                    return 1 + transfers(net_balances[1:i] + net_balances[i + 1:])
            min_transfers = float("inf")
            for i in range(1, len(net_balances)):
                if b * net_balances[i] < 0:         
                    count = 1 + transfers(net_balances[1:i] + net_balances[i + 1:] + [b + net_balances[i]])
                    min_transfers = min(min_transfers, count)
            return min_transfers
        return transfers(net_balances)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def isOneEditDistance(self, s, t):
        diff = len(s) - len(t)
        if abs(diff) > 1:       
            return False
        edit = False
        if diff == 0:           
            for c_s, c_t in zip(s, t):
                if c_s != c_t:
                    if edit:    
                        return False
                    edit = True
            return edit         
        else:                   
            long, short = s, t
            if diff < 0:
                long, short = short, long
            i = 0               
            while i < len(short) and long[i] == short[i]:
                i += 1
            return long[i+1:] == short[i:]  
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findAnagrams(self, s, p):
        n = len(p)
        freq = defaultdict(int)     
        result = []
        if n > len(s):              
            return result
        def update_freq(c, step):   
            freq[c] += step
            if freq[c] == 0:
                del freq[c]
        for c1, c2 in zip(p, s[:n]):    
            update_freq(c1, -1)
            update_freq(c2, 1)
        for i in range(len(s) - n):
            if not freq:
                result.append(i)
            update_freq(s[i], -1)       
            update_freq(s[i + n], 1)    
        if not freq:
            result.append(len(s) - n)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def validateStackSequences(self, pushed, popped):
        stack = []
        i = 0                   
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[i]:
                i += 1
                stack.pop()
        return i == len(popped) 
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def validMountainArray(self, A):
        n = len(A)
        left, right = 0, n - 1
        while left + 1 < n - 1 and A[left + 1] > A[left]:
            left += 1
        while right - 1 > 0 and A[right - 1] > A[right]:
            right -= 1
        return 0 < left == right < n - 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def arrangeCoins(self, n):
        return int(math.sqrt(1 + 8.0 * n) - 1) / 2
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def robotSim(self, commands, obstacles):
        NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]     
        position, orientation = (0, 0), NORTH
        max_sqr_distance = 0
        obstacles = {tuple(obstacle) for obstacle in obstacles}  
        for command in commands:
            if command == -2:
                orientation = (orientation - 1) % 4         
            elif command == -1:
                orientation = (orientation + 1) % 4         
            else:
                for _ in range(command):
                    next_position = (position[0] + directions[orientation][0],
                                     position[1] + directions[orientation][1])
                    if next_position in obstacles:          
                        break
                    position = next_position
                    max_sqr_distance = max(max_sqr_distance, position[0] ** 2 + position[1] ** 2)
        return max_sqr_distance
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def rearrangeString(self, s, k):
        freq = Counter(s)
        heap = [(-count, letter) for letter, count in freq.items()]
        heapq.heapify(heap)
        last_used = {}      
        rearranged = []
        while heap:
            too_close = []          
            neg_f, letter = heapq.heappop(heap)
            while letter in last_used and len(rearranged) - last_used[letter] < k:  
                too_close.append((neg_f, letter))
                if not heap:    
                    return ""
                neg_f, letter = heapq.heappop(heap)
            last_used[letter] = len(rearranged)
            rearranged.append(letter)
            neg_f += 1
            if neg_f:               
                heapq.heappush(heap, (neg_f, letter))
            for item in too_close:  
                heapq.heappush(heap, item)
        return "".join(rearranged)
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep):
    while node:
        print(node.data, end='')
        node = node.next
        if node:
            print(sep, end='')
def reversePrint(head):
    if not head:
        return None
    reversePrint(head.next)
    print(head.data)
if __name__ == '__main__':
    tests = int(input())
    for tests_itr in range(tests):
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        reversePrint(llist.head)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def constructRectangle(self, area):
        side = int(area ** 0.5)
        while area % side != 0:         
            side -= 1
        return [area // side, side]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numberOfLines(self, widths, S):
        line, width = 0, 100            
        for c in S:
            c_length = widths[ord(c) - ord("a")]
            if width + c_length > 100:  
                line += 1
                width = 0
            width += c_length
        return [line, width]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def rotateRight(self, head, k):
        if not head:
            return
        count = 1
        node = head
        while node.next:
            node = node.next
            count += 1
        node.next = head        
        to_move = count - (k % count)   
        while to_move > 0:
            node = node.next
            to_move -= 1
        head = node.next                
        node.next = None
        return head
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxSumOfThreeSubarrays(self, nums, k):
        one_sum = sum(nums[:k])                 
        two_sum = sum(nums[k:k * 2])
        three_sum = sum(nums[k * 2:k * 3])
        best_one = one_sum                      
        best_two = one_sum + two_sum
        best_three = one_sum + two_sum + three_sum
        best_one_i = 0                          
        best_two_i = [0, k]
        best_three_i = [0, k, k * 2]
        one_i = 1                               
        two_i = k + 1
        three_i = k * 2 + 1
        while three_i <= len(nums) - k:
            one_sum += nums[one_i + k - 1] - nums[one_i - 1]    
            two_sum += nums[two_i + k - 1] - nums[two_i - 1]
            three_sum += nums[three_i + k - 1] - nums[three_i - 1]
            if one_sum > best_one:                  
                best_one = one_sum
                best_one_i = one_i
            if best_one + two_sum > best_two:       
                best_two = best_one + two_sum
                best_two_i = [best_one_i, two_i]
            if best_two + three_sum > best_three:   
                best_three = best_two + three_sum
                best_three_i = best_two_i + [three_i]
            one_i += 1
            two_i += 1
            three_i += 1
        return best_three_i
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def containVirus(self, grid):
        rows, cols = len(grid), len(grid[0])
        used_walls = 0
        def get_nbors(r, c):    
            if (r, c) in visited:
                return
            visited.add((r, c))
            for dr, dc in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                r1, c1 = r + dr, c + dc
                if r1 < 0 or r1 >= rows or c1 < 0 or c1 >= cols:
                    continue
                if grid[r1][c1] == 2:       
                    continue
                if grid[r1][c1] == 0:       
                    nbors.add((r1, c1))
                    walls[0] += 1
                else:
                    get_nbors(r1, c1)       
        def contain_region(r, c):   
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return
            if grid[r][c] != 1:
                return
            grid[r][c] = 2
            for dr, dc in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                contain_region(r + dr, c + dc)
        while True:
            regions = []
            visited = set()
            for r in range(rows):
                for c in range(cols):
                    if (r, c) not in visited and grid[r][c] == 1:
                        nbors, walls = set(), [0]
                        get_nbors(r, c)
                        regions.append([(r, c), set(nbors), walls[0]])
            regions.sort(key = lambda x: -len(x[1]))        
            if not regions or len(regions[0][1]) == 0:      
                return used_walls
            used_walls += regions[0][2]                     
            contain_region(regions[0][0][0], regions[0][0][1])
            for _, expansion, _ in regions[1:]:             
                for r, c in expansion:
                    grid[r][c] = 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
class Solution(object):
    def findRightInterval(self, intervals):
        intervals = [[intervals[i], i] for i in range(len(intervals))]
        intervals.sort(key=lambda x: x[0].start)
        result = [-1] * len(intervals)      
        for interval, i in intervals:
            left, right = 0, len(intervals) 
            while left < right:
                mid = (left + right) // 2
                if intervals[mid][0].start < interval.end:
                    left = mid + 1
                else:
                    right = mid
            if left == len(intervals):
                continue
            result[i] = intervals[left][1]
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numFactoredBinaryTrees(self, A):
        MOD = 10 ** 9 + 7
        num_to_trees = Counter(A)       
        A.sort()
        for i, num in enumerate(A):
            for left in A[:i]:
                right, remainder = divmod(num, left)
                if right <= 1:
                    break
                if remainder == 0 and right in num_to_trees:
                    num_to_trees[num] += num_to_trees[left] * num_to_trees[right]
        return sum(num_to_trees.values()) % MOD
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def poorPigs(self, buckets, minutesToDie, minutesToTest):
        rounds = minutesToTest // minutesToDie
        return int(ceil(log(buckets) / log(rounds + 1)))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numRabbits(self, answers):
        colours = {}                            
        rabbits = 0
        for rabbit in answers:
            if colours.get(rabbit, 0) > 0:      
                colours[rabbit] -= 1
            else:
                rabbits += rabbit + 1           
                colours[rabbit] = rabbit
        return rabbits
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def threeSumMulti(self, A, target):
        counts = [0] * 101
        for num in A:
            counts[num] += 1
        result = 0
        for small, small_count in enumerate(counts):
            if small_count == 0:
                continue
            for med, med_count in enumerate(counts[small:], small):
                if med_count == 0:
                    continue
                other = target - small - med
                if other < 0 or other > 100 or counts[other] == 0:
                    continue
                other_count = counts[other]
                if small == med == other:
                    result += small_count * (small_count - 1) * (small_count - 2) // 6
                elif small == med:          
                    result += small_count * (small_count - 1) * other_count // 2
                elif other > med:           
                    result += small_count * med_count * other_count
        return result % (10 ** 9 + 7)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def totalNQueens(self, n):
        partials = [[]]
        for col in range(n):
            new_partials = []
            for partial in partials:
                for row in range(n):
                    if not self.conflict(partial, row):
                        new_partials.append(partial + [row])
            partials = new_partials
        return len(partials)
    def conflict(self, partial, new_row):
        for col, row in enumerate(partial):
            if new_row == row:
                return True
            col_diff = len(partial) - col
            if row + col_diff == new_row or row - col_diff == new_row:
                return True
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def countTriplets(self, A):
        pairs = defaultdict(int)
        for num1 in A:
            for num2 in A:
                pairs[num1 & num2] += 1
        result = 0
        for pair, count in pairs.items():
            for num3 in A:
                if pair & num3 == 0:
                    result += count
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def combinationSum3(self, k, n):
        results = []
        self.cs3([], n, results, k)
        return results
    def cs3(self, partial, target, results, k):
        if len(partial) == k and target == 0:   
            results.append(partial)
        if len(partial) >= k or target <= 0:    
            return
        last_used = 0 if not partial else partial[-1]
        for i in range(last_used+1, 10):        
            self.cs3(partial + [i], target-i, results, k)
EOF
class Solution(object):
    def PredictTheWinner(self, nums):
        if len(nums) % 2 == 0 or len(nums) == 1:
            return True
        dp = [0] * len(nums)
        for i in reversed(xrange(len(nums))):
            dp[i] = nums[i]
            for j in xrange(i+1, len(nums)):
                dp[j] = max(nums[i] - dp[j], nums[j] - dp[j - 1])
        return dp[-1] >= 0
EOF
class Solution(object):
    def isRectangleOverlap(self, rec1, rec2):
        def intersect(p_left, p_right, q_left, q_right):
            return max(p_left, q_left) < min(p_right, q_right)
        return (intersect(rec1[0], rec1[2], rec2[0], rec2[2]) and
                intersect(rec1[1], rec1[3], rec2[1], rec2[3]))
EOF
class Solution(object):
    def findTheDifference(self, s, t):
        return chr(reduce(operator.xor, map(ord, s), 0) ^ reduce(operator.xor, map(ord, t), 0))
    def findTheDifference2(self, s, t):
        t = list(t)
        s = list(s)
        for i in s:
            t.remove(i)
        return t[0]
    def findTheDifference3(self, s, t):
        return chr(reduce(operator.xor, map(ord, s + t)))
    def findTheDifference4(self, s, t):
        return list((collections.Counter(t) - collections.Counter(s)))[0]
    def findTheDifference5(self, s, t):
        s, t = sorted(s), sorted(t)
        return t[-1] if s == t[:-1] else [x[1] for x in zip(s, t) if x[0] != x[1]][0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def trimBST(self, root, L, R):
        if not root:
            return None
        if root.val > R:
            return self.trimBST(root.left, L, R)
        if root.val < L:
            return self.trimBST(root.right, L, R)
        root.left = self.trimBST(root.left, L, R)
        root.right = self.trimBST(root.right, L, R)
        return root
EOF
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        for i in xrange(len(flowerbed)):
            if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) and \
                (i == len(flowerbed)-1 or flowerbed[i+1] == 0):
                flowerbed[i] = 1
                n -= 1
            if n <= 0:
                return True
        return False
EOF
for i in range(1,int(input())+1): 
    print((10**i//9)**2)
EOF
if __name__ == "__main__":
    num = int(input().strip())
    history = OrderedDict()
    for _ in range(num):
        word = str(input().strip().split())
        if word not in history.keys():
            history[word] = 1
        else:
            history[word] += 1
    print(len(history.keys()))
    print(" ".join(map(str, history.values())))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        prev = dummy = ListNode(None)       
        while l1 and l2:                    
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 or l2                
        return dummy.next
EOF
maxdepth = 0
def depth(elem, level):
    if level == -1:
        level = 0
    global maxdepth
    if level > maxdepth:
        maxdepth = level
    for el in elem:
        depth(el, level+1)
EOF
def aVeryBigSum(ar):
    summ = 0
    for i in ar:
        summ += i
    return summ 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = aVeryBigSum(ar)
    fptr.write(str(result) + '\n')
    fptr.close()
EOF
def binomial(x, y):
    if y == x:
        return 1
    elif y == 1:         
        return x
    elif y > x:          
        return 0
    else:                
        a = math.factorial(x)
        b = math.factorial(y)
        c = math.factorial(x-y)  
        div = a // (b * c)
        return div
def get_all_substrings(input_string):
    length = len(input_string)
    return [input_string[i:j+1] for i in range(length) for j in range(i,length)]
def sherlockAndAnagrams(s):
    res = 0
    handict = defaultdict(int)
    for sub in get_all_substrings(s):
        handict[str(sorted(sub))] += 1
    for el in list(filter(lambda x: x[1] != 1, handict.items())):
        res += binomial(el[1], 2)
    return res
q = int(input().strip())
for a0 in range(q):
    s = input().strip()
    result = sherlockAndAnagrams(s)
    print(result)
EOF
class Solution(object):
    def numUniqueEmails(self, emails):
        def convert(email):
            name, domain = email.split('            name = name[:name.index('+')]
            return "".join(["".join(name.split(".")), '
        lookup = set()
        for email in emails:
            lookup.add(convert(email))
        return len(lookup)
EOF
class Solution(object):
    def nextClosestTime(self, time):
        h, m = time.split(":")
        curr = int(h) * 60 + int(m)
        result = None
        for i in xrange(curr+1, curr+1441):
            t = i % 1440
            h, m = t // 60, t % 60
            result = "%02d:%02d" % (h, m)
            if set(result) <= set(time):
                break
        return result
EOF
class Solution(object):
    def maxSatisfaction(self, satisfaction):
        satisfaction.sort(reverse=True)
        result, curr = 0, 0
        for x in satisfaction:
            curr += x
            if curr <= 0:
                break
            result += curr
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numDecodings(self, s):
        if not s:
            return 0
        nb_ways = [0] * (len(s)+1)      
        nb_ways[0] = 1                  
        if s[0] != '0':
            nb_ways[1] = 1
        for i in range(1, len(s)):
            if s[i] != '0':                     
                nb_ways[i+1] += nb_ways[i]      
            if 10 <= int(s[i-1:i+1]) <= 26:     
                nb_ways[i+1] += nb_ways[i-1]    
        return nb_ways[-1]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeLinkNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None
class Solution(object):
    def connect(self, root):
        if not root:
            return
        level = [root]
        while level:
            next_level = []
            prev = None
            for node in level:
                if prev:
                    prev.next = node
                prev = node
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            level = next_level
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def toGoatLatin(self, S):
        S = S.split()
        vowels = {"a", "e", "i", "o", "u"}
        for i, word in enumerate(S):
            if word[0].lower() not in vowels:
                S[i] = S[i][1:] + S[i][0]
            S[i] += "ma" + "a" * (i + 1)
        return " ".join(S)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countRangeSum(self, nums, lower, upper):
        cumul = [0]
        for num in nums:
            cumul.append(num + cumul[-1])
        def mergesort(cumul, left, right):      
            count = 0
            if right - left <= 1:
                return count
            mid = (left + right) // 2
            count += mergesort(cumul, left, mid) + mergesort(cumul, mid, right)     
            i, j = mid, mid
            for prefix_sum in cumul[left:mid]:                      
                while i < right and cumul[i] - prefix_sum < lower:  
                    i += 1
                while j < right and cumul[j] - prefix_sum <= upper: 
                    j += 1
                count += (j - i)    
            cumul[left:right] = sorted(cumul[left:right])   
            return count
        return mergesort(cumul, 0, len(cumul))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestPathAllKeys(self, grid):
        rows, cols = len(grid), len(grid[0])
        possible_keys = set("abcdef")
        keys = set()
        for r in range(rows):                   
            for c in range(cols):
                if grid[r][c] == "                    start_r, start_c = r, c
                elif grid[r][c] in possible_keys:
                    keys.add(grid[r][c])
        steps = 0
        frontier = [(start_r, start_c, "")]     
        visited = set()
        neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while frontier:
            new_frontier = set()
            for r, c, open_locks in frontier:
                if (r, c, open_locks) in visited:    
                    continue
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    continue
                if grid[r][c] == "
                    continue
                if "A" <= grid[r][c] <= "F" and grid[r][c] not in open_locks:
                    continue
                visited.add((r, c, open_locks))
                if grid[r][c] in keys and grid[r][c].upper() not in open_locks:
                    open_locks = "".join(sorted(open_locks + grid[r][c].upper()))  
                if len(open_locks) == len(keys):
                    return steps
                for dr, dc in neighbours:
                    new_frontier.add((r + dr, c + dc, open_locks))
            frontier = new_frontier
            steps += 1
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        result = [-1] * len(nums1)                      
        find_to_i = {}                                  
        for i, num in enumerate(nums1):
            find_to_i[num] = i
        stack = []
        for num in nums2:
            while stack and num > stack[-1]:            
                smaller = stack.pop()
                if smaller in find_to_i:
                    result[find_to_i[smaller]] = num
            stack.append(num)
        return result 
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def generate(self, numRows):
        if numRows == 0:
            return []
        pascal = [[1]]
        for i in range(1, numRows):
            pascal.append([1])
            for num1, num2 in zip(pascal[-2][:-1], pascal[-2][1:]):
                pascal[-1].append(num1 + num2)
            pascal[-1].append(1)
        return pascal
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def alienOrder(self, words):
        after = defaultdict(int)    
        order = defaultdict(set)    
        seen = set(words[0])        
        for i in range(1, len(words)):
            diff_to_prev = False
            for j, c in enumerate(words[i]):
                seen.add(c)         
                if j < len(words[i-1]) and not diff_to_prev and c != words[i-1][j]:
                    if c not in order[words[i-1][j]]:   
                        order[words[i-1][j]].add(c)
                        after[c] += 1
                    diff_to_prev = True
            if not diff_to_prev and len(words[i-1]) > len(words[i]):    
                return ""
        for c in seen:              
            if c not in after:
                after[c] = 0
        frontier = set()            
        for a in after:
            if after[a] == 0:
                frontier.add(a)
        letters = []
        while frontier:
            b = frontier.pop()      
            del after[b]
            letters.append(b)
            for a in order[b]:      
                after[a] -= 1
                if after[a] == 0:
                    frontier.add(a)
        if after:
            return ""
        return "".join(letters)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def minDeletionSize(self, A):
        deletions = 0
        for col in zip(*A):     
            if list(col) != sorted(col):
                deletions += 1
        return deletions
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findLonelyPixel(self, picture):
        pixels = 0
        rows, cols = len(picture), len(picture[0])
        col_counts = [0 for _ in range(cols)]  
        row_pixels = [[] for _ in range(rows)]  
        for r in range(rows):
            for c in range(cols):
                if picture[r][c] == "B":
                    col_counts[c] += 1
                    row_pixels[r].append(c)
        for r in range(rows):
            if len(row_pixels[r]) == 1:
                c = row_pixels[r][0]
                if col_counts[c] == 1:
                    pixels += 1
        return pixels
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getPermutation(self, n, k):
        chars = [str(i) for i in range(1, n+1)]     
        permutations = factorial(n)                 
        k -= 1                                      
        result = []
        while chars:
            digit = n * k // permutations           
            result.append(chars[digit])             
            del chars[digit]                        
            permutations //= n                      
            k -= digit * permutations
            n -= 1
        return "".join(result)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def partition(self, s):
        partitons = []
        self.find_partitions(s, [], partitons)
        return partitons
    def find_partitions(self, s, partial, partitions):
        if not s:
            partitions.append(partial)
        for i in range(1, len(s)+1):
            prefix = s[:i]
            if prefix == prefix[::-1]:
                self.find_partitions(s[i:], partial + [s[:i]], partitions)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def movesToStamp(self, stamp, target):
        memo = {}
        def helper(i, j, results):          
            if (i, j) in memo:
                return memo[(i, j)]
            if len(results) > 10 * len(target): 
                return []
            if i == len(target):            
                return results if j == len(stamp) else []
            if j == len(stamp):             
                for k in range(len(stamp)):
                    temp = helper(i, k, [i - k] + results)  
                    if temp:
                        result = temp
                        break
                else:
                    result = []
            elif target[i] != stamp[j]:     
                result = []
            else:
                temp = helper(i + 1, j + 1, results)    
                if temp:
                    result = temp
                else:                       
                    result = helper(i + 1, 0, results + [i + 1])
            memo[(i, j)] = result
            return result
        return helper(0, 0, [0])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def addOperators(self, num, target):
        if not num:
            return []
        self.num, self.target, self.expressions = num, target, []
        self.helper("", 0, 0, 0)
        return self.expressions
    def helper(self, path, index, eval, multed):
        if index == len(self.num) and self.target == eval:
            self.expressions.append(path)
        for i in range(index, len(self.num)):   
            if i != index and self.num[index] == '0':
                break                           
            cur_str = self.num[index:i+1]       
            cur_int = int(cur_str)
            if index == 0:                      
                self.helper(path + cur_str, i + 1, cur_int, cur_int)
            else:
                self.helper(path + "+" + cur_str, i + 1, eval + cur_int , cur_int)
                self.helper(path + "-" + cur_str, i + 1, eval - cur_int, -cur_int)
                self.helper(path + "*" + cur_str, i + 1, eval - multed + multed * cur_int, multed * cur_int)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numberOfBoomerangs(self, points):
        def dist_squared(p1, p2):
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        boomerangs = 0
        for middle in points:
            distances = defaultdict(int)
            for i, other in enumerate(points):
                distances[dist_squared(middle, other)] += 1
            for count in distances.values():
                boomerangs += count * (count - 1)
        return boomerangs
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def restoreIpAddresses(self, s):
        NB_SECTIONS = 4
        if 3 * NB_SECTIONS < len(s) < NB_SECTIONS:    
            return []
        results = [[]]
        while NB_SECTIONS > 0:
            new_results = []
            for result in results:
                used = sum((len(section) for section in result))    
                remaining = len(s) - used                           
                if 3 * (NB_SECTIONS - 1) >= remaining - 3 >= NB_SECTIONS - 1 and 100 <= int(s[used:used + 3]) <= 255:
                    new_results.append(result + [s[used:used + 3]])
                if 3 * (NB_SECTIONS - 1) >= remaining - 2 >= NB_SECTIONS - 1 and 10 <= int(s[used:used + 2]):
                    new_results.append(result + [s[used:used + 2]])
                if 3 * (NB_SECTIONS - 1) >= remaining - 1 >= NB_SECTIONS - 1:
                    new_results.append(result + [s[used]])
            NB_SECTIONS -= 1
            results = new_results
        return ['.'.join(result) for result in results]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def detectCycle(self, head):
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                fast = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow
        return None
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def removeBoxes(self, boxes):
        def helper(left, right, same):  
            if left > right:
                return 0
            if (left, right, same) in memo:
                return memo[(left, right, same)]
            while right > left and boxes[right] == boxes[right - 1]:
                right -= 1
                same += 1
            result = helper(left, right - 1, 0) + (same + 1) ** 2  
            for i in range(left, right):    
                if boxes[i] == boxes[right]:
                    result = max(result, helper(i + 1, right - 1, 0) + helper(left, i, same + 1))
            memo[(left, right, same)] = result
            return result
        memo = {}
        return helper(0, len(boxes) - 1, 0)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class WordDistance(object):
    def __init__(self, words):
        self.word_indices = defaultdict(list)
        for i, word in enumerate(words):
            self.word_indices[word].append(i)
    def shortest(self, word1, word2):
        i1 = self.word_indices[word1]   
        i2 = self.word_indices[word2]   
        distance = float('inf')
        p1, p2 = 0, 0                   
        while p1 < len(i1) and p2 < len(i2):    
            distance = min(distance, abs(i1[p1] - i2[p2]))
            if i1[p1] < i2[p2]:     
                p1 += 1
            else:
                p2 += 1
        return distance
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numMagicSquaresInside(self, grid):
        def is_magic(row, col):
            if grid[row + 1][col + 1] != 5:     
                return False
            line_sums = [0 for _ in range(6)]   
            diag1, diag2 = 0, 0                 
            for dr in range(3):
                for dc in range(3):
                    val = grid[row + dr][col + dc]
                    if val < 1 or val > 9:      
                        return False
                    line_sums[dr] += val        
                    line_sums[dc + 3] += val
                    if dr == dc:
                        diag1 += val
                    if dr + dc == 2:
                        diag2 += val
            if any(line_sum != 15 for line_sum in line_sums):
                return False
            if diag1 != 15 or diag2 != 15:
                return False
            return True
        rows, cols = len(grid), len(grid[0])
        magic = 0
        for r in range(rows - 2):
            for c in range(cols - 2):
                magic += is_magic(r, c)
        return magic
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countCornerRectangles(self, grid):
        rows, cols = len(grid), len(grid[0])
        cols_by_row = []
        for r in range(rows):
            cols_by_row.append(set((c for c in range(cols) if grid[r][c])))
        rectangles = 0
        for high_row in range(1, rows):
            for low_row in range(high_row):
                common_cols = len(cols_by_row[high_row] & cols_by_row[low_row])
                if common_cols >= 2:
                    rectangles += common_cols * (common_cols - 1) // 2
        return rectangles
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def kthSmallest(self, matrix, k):
        rows, cols = len(matrix), len(matrix[0])
        if k > (rows * cols) // 2:
            back = True
            k = rows * cols - k + 1
            frontier = [(-matrix[rows - 1][cols - 1], rows - 1, cols - 1)]
        else:
            back = False
            frontier = [(matrix[0][0], 0, 0)]
        while k:
            val, r, c = heapq.heappop(frontier)
            k -= 1
            if not back:
                if c != len(matrix[0]) - 1:
                    heapq.heappush(frontier, (matrix[r][c + 1], r, c + 1))
                if c == 0 and r != len(matrix) - 1:
                    heapq.heappush(frontier, (matrix[r + 1][c], r + 1, c))
            else:
                if c != 0:
                    heapq.heappush(frontier, (-matrix[r][c - 1], r, c - 1))
                if c == cols - 1 and r != 0:
                    heapq.heappush(frontier, (-matrix[r - 1][c], r - 1, c))
        return -val if back else val
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def firstUniqChar(self, s):
        counts = [0 for _ in range(26)]
        for c in s:
            counts[ord(c) - ord("a")] += 1
        for i, c in enumerate(s):
            if counts[ord(c) - ord("a")] == 1:
                return i
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def topKFrequent(self, nums, k):
        n = len(nums)
        frequencies = [[] for _ in range(n + 1)]
        for num, freq in Counter(nums).items():
            frequencies[freq].append(num)
        top_k = []
        while k:
            while not frequencies[n]:
                n -= 1
            top_k.append(frequencies[n].pop())
            k -= 1
        return top_k
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def myAtoi(self, str):
        str = str.strip()       
        negative = False        
        if str and str[0] == '-':
            negative = True
        if str and (str[0] == '+' or str[0] == '-'):
            str = str[1:]
        if not str:
            return 0
        digits = {i for i in '0123456789'}
        result = 0
        for c in str:           
            if c not in digits:
                break
            result = result * 10 + int(c)
        if negative:
            result = -result
        result = max(min(result, 2**31 - 1), -2**31)    
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]:
            return False
        rows, cols = len(matrix), len(matrix[0])
        low, high = 0, rows * cols - 1
        while high >= low:
            mid = (high + low) // 2
            value = matrix[mid // cols][mid % cols]     
            if target == value:
                return True
            if target > value:
                low = mid + 1
            else:
                high = mid - 1
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def reverseBetween(self, head, m, n):
        pseudo = ListNode(None)     
        pseudo.next = head
        node = pseudo
        n -= m                      
        while m > 1:                
            node = node.next
            m -= 1
        reversed_head = None
        next_reverse = node.next
        while n >= 0:               
            tail = next_reverse.next
            next_reverse.next = reversed_head
            reversed_head = next_reverse
            next_reverse = tail
            n -= 1
        node.next.next = tail       
        node.next = reversed_head   
        return pseudo.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def find132pattern(self, nums):
        two = float("-inf")
        stack = []
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] < two:
                return True
            while stack and stack[-1] < nums[i]:
                two = stack.pop()
            stack.append(nums[i])
        return False
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def getMoneyAmount(self, n):
        min_money = [[0 for _ in range(n)], [i for i in range(1, n)]]
        for range_length in range(3, n + 1):
            min_money.append([])
            for lower in range(1, n + 2 - range_length):
                upper = lower + range_length - 1
                min_cost = float('inf')
                for guess in range((lower + upper)  // 2, upper):   
                    cost = guess + max(min_money[guess - lower - 1][lower - 1], min_money[upper - guess - 1][guess])
                    min_cost = min(min_cost, cost)
                min_money[-1].append(min_cost)
        return min_money[n - 1][0]
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minDeletionSize(self, A):
        rows, cols = len(A), len(A[0])
        max_subsequence = [1] * cols
        for col_end in range(1, cols):
            for col in range(col_end):
                if all(A[r][col] <= A[r][col_end] for r in range(rows)):
                    max_subsequence[col_end] = max(max_subsequence[col_end],
                                                   max_subsequence[col] + 1)
        return cols - max(max_subsequence)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def fallingSquares(self, positions):
        box_heights = [positions[0][1]]  
        max_heights = [positions[0][1]]
        for left, side in positions[1:]:
            top = side  
            for i in range(len(box_heights)):  
                left2, side2 = positions[i]
                if left2 < left + side and left2 + side2 > left:  
                    top = max(top, box_heights[i] + side)  
            box_heights.append(top)
            max_heights.append(max(top, max_heights[-1]))
        return max_heights
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
class Solution(object):
    def copyRandomList(self, head):
        node = head
        while node:
            next = node.next
            copy = RandomListNode(node.label)
            node.next = copy
            copy.next = next
            node = next
        node = head
        while node:
            if node.random:
                node.next.random = node.random.next
            node = node.next.next
        pseudo = prev = RandomListNode(0)
        node = head
        while node:
            prev.next = node.next
            node.next = node.next.next
            node = node.next
            prev = prev.next
        return pseudo.next
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wallsAndGates(self, rooms):
        if not rooms or not rooms[0]:
            return
        INF = 2 ** 31 - 1
        rows, cols = len(rooms), len(rooms[0])
        frontier = deque([(r, c) for r in range(rows) for c in range(cols) if rooms[r][c] == 0])
        while frontier:
            row, col = frontier.popleft()
            for i, j in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                if i >= 0 and i < rows and j >= 0 and j < cols:
                    if rooms[i][j] == INF:
                        rooms[i][j] = rooms[row][col] + 1
                        frontier.append((i, j))
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findNumberOfLIS(self, nums):
        if not nums:
            return 0
        lengths, counts = [], []
        for i, num in enumerate(nums):
            length, count = 1, 1    
            for j in range(i):
                if num > nums[j]:                   
                    if lengths[j] + 1 > length:     
                        length = lengths[j] + 1
                        count = counts[j]
                    elif lengths[j] + 1 == length:  
                        count += counts[j]          
            lengths.append(length)
            counts.append(count)
        longest = max(lengths)
        return sum([count for length, count in zip(lengths, counts) if length == longest])
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def climbStairs(self, n):
        if n <= 0:
            return 0
        if n <= 2:
            return n
        stairs, prev = 2, 1         
        for _ in range(3, n + 1):
            stairs, prev = stairs + prev, stairs
        return stairs
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def maxPathSum(self, root):
        return self.helper(root)[0]
    def helper(self, node):     
        if not node:
            return float('-inf'), 0     
        left_via, left_down = self.helper(node.left)
        right_via, right_down = self.helper(node.right)
        via = max(node.val + max(0, left_down) + max(0, right_down), left_via, right_via)
        down = node.val + max(0, left_down, right_down)
        return via, down
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def detectCapitalUse(self, word):
        if len(word) <= 1:          
            return True
        first = word[0] <= "Z"
        second = word[1] <= "Z"
        if not first and second:    
            return False
        for c in word[2:]:
            if (c <= "Z") != second:
                return False
        return True
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lenLongestFibSubseq(self, A):
        A_set = set(A)
        max_length = 0
        for i, num in enumerate(A):
            for num2 in A[i + 1:]:
                prev_num = num2
                next_num = num + num2
                length = 2
                while next_num in A_set:    
                    length += 1
                    next_num, prev_num = next_num + prev_num, next_num
                max_length = max(max_length, length)
        return max_length if max_length >= 3 else 0
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class BSTIterator(object):
    def __init__(self, root):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left
    def hasNext(self):
        return True if self.stack else False
    def next(self):
        node = self.stack.pop()
        result = node.val
        if node.right:
            node = node.right
            while node:
                self.stack.append(node)
                node = node.left
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def binaryGap(self, N):
        previous, max_gap = None, 0         
        i = 0                               
        while N > 0:
            if N & 1:                       
                if previous is not None:
                    max_gap = max(max_gap, i - previous)
                previous = i
            N >>= 1                         
            i += 1
        return max_gap
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def wiggleSort(self, nums):
        nums.sort()
        median = nums[len(nums)//2]
        def mapping(i):
            return (i*2 + 1) % (len(nums) | 1)      
        left, i, right = 0, 0, len(nums) - 1
        while i <= right:
            if nums[mapping(i)] > median:
                nums[mapping(i)], nums[mapping(left)] = nums[mapping(left)], nums[mapping(i)]
                left += 1
                i += 1
            elif nums[mapping(i)] < median:
                nums[mapping(i)], nums[mapping(right)] = nums[mapping(right)], nums[mapping(i)]
                right -= 1
            else:
                i += 1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution(object):
    def reorderList(self, head):
        if not head:
            return None
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        prev, node = None, slow
        while node:
            prev, node.next, node = node, prev, node.next
        first, second = head, prev  
        while second.next:
            first.next, first = second, first.next      
            second.next, second = first, second.next    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def minMutation(self, start, end, bank):
        chars = set("ACGT")
        bank = set(bank)        
        if end not in bank:     
            return -1
        distance = 0
        frontier = [start]
        while frontier:
            new_frontier = []
            distance += 1
            for gene in frontier:
                for i in range(len(gene)):
                    for c in chars:
                        if c == gene[i]:
                            continue
                        mutation = list(gene)           
                        mutation[i] = c
                        mutation = "".join(mutation)
                        if mutation == end:             
                            return distance
                        if mutation in bank:
                            bank.discard(mutation)      
                            new_frontier.append(mutation)
            frontier = new_frontier
        return -1
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def oddEvenJumps(self, A):
        n = len(A)
        def next_list():                                
            result = [None] * n
            stack = []
            for i in indices:
                while stack and i > stack[-1]:          
                    result[stack.pop()] = i             
                stack.append(i)
            return result
        indices = sorted(range(n), key=lambda x: A[x])  
        next_larger = next_list()
        indices.sort(key=lambda x: -A[x])               
        next_smaller = next_list()
        odd = [False] * (n - 1) + [True]                
        even = [False] * (n - 1) + [True]
        for i in range(n - 2, -1, -1):                  
            if next_larger[i] is not None:
                odd[i] = even[next_larger[i]]
            if next_smaller[i] is not None:
                even[i] = odd[next_smaller[i]]
        return sum(odd)                                 
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def longestConsecutive(self, root):
        self.longest = 0
        self.consecutive(root, float('inf'), 0)
        return self.longest
    def consecutive(self, node, parent_val, sequence):
        if not node:
            return
        if node.val == 1 + parent_val:
            sequence += 1
        else:
            sequence = 1
        self.longest = max(self.longest, sequence)
        self.consecutive(node.left, node.val, sequence)
        self.consecutive(node.right, node.val, sequence)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def cherryPickup(self, grid):
        n = len(grid)
        memo = {}
        def helper(r1, c1, r2):
            c2 = r1 + c1 - r2
            if r1 == n or c1 == n or r2 == n or c2 == n:    
                return float("-inf")
            if grid[r1][c1] == -1 or grid[r2][c2] == -1:    
                return float("-inf")
            if r1 == n - 1 and c1 == n - 1:                 
                return grid[n - 1][n - 1]
            if (r1, c1, r2) in memo:
                return memo[(r1, c1, r2)]
            result = grid[r1][c1]                           
            if r2 != r1 or c2 != c1:                        
                result += grid[r2][c2]                      
            result += max(helper(r1 + 1, c1, r2 + 1), helper(r1, c1 + 1, r2),
                          helper(r1 + 1, c1, r2), helper(r1, c1 + 1, r2 + 1))
            memo[(r1, c1, r2)] = result
            return result
        return max(0, helper(0, 0, 0))                      
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def firstBadVersion(self, n):
        left, right = 1, n          
        while left < right:
            mid = (left + right) // 2
            if isBadVersion(mid):   
                right = mid
            else:                   
                left = mid + 1
        return left
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def shortestPathLength(self, graph):
        if len(graph) == 0 or len(graph[0]) == 0:
            return 0
        n = len(graph)
        frontier = {(1 << node, node) for node in range(n)}     
        visited = set(frontier)
        distance = 0
        while True:
            new_frontier = set()
            for bit_nodes, node in frontier:
                if bit_nodes == 2 ** n - 1:                     
                    return distance
                for nbor in graph[node]:
                    new_bit_nodes = bit_nodes | 1 << nbor       
                    if (new_bit_nodes, nbor) not in visited:
                        new_frontier.add((new_bit_nodes, nbor))
            visited |= new_frontier                             
            distance += 1
            frontier = new_frontier
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def largestDivisibleSubset(self, nums):
        max_to_set = {-1 : set()}   
        nums.sort()
        for num in nums:
            num_set = set()         
            for max_in_s, s in max_to_set.items():
                if num % max_in_s == 0 and len(s) > len(num_set):
                    num_set = s
            max_to_set[num] = num_set | {num}   
        return list(max(max_to_set.values(), key = len))    
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def singleNumber(self, nums):
        ones, twos = 0, 0
        for num in nums:
            ones = (ones ^ num) & ~twos     
            twos = (twos ^ num) & ~ones     
        return ones
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def pacificAtlantic(self, matrix):
        if not matrix or not matrix[0]:
            return []
        rows, cols = len(matrix), len(matrix[0])
        atlantic, pacific = set(), set()        
        for r in range(rows):
            atlantic.add((r, cols - 1))
            pacific.add((r, 0))
        for c in range(cols):
            atlantic.add((rows - 1, c))
            pacific.add((0, c))
        for ocean in [atlantic, pacific]:
            frontier = set(ocean)
            while frontier:
                new_frontier = set()
                for r, c in frontier:
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        r1, c1 = r + dr, c + dc     
                        if r1 < 0 or r1 >= rows or c1 < 0 or c1 >= cols or (r1, c1) in ocean:
                            continue                
                        if matrix[r1][c1] >= matrix[r][c]:  
                            new_frontier.add((r1, c1))
                frontier = new_frontier             
                ocean |= new_frontier               
        return list(atlantic & pacific)             
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def findKthNumber(self, n, k):
        kth = 1
        k -= 1
        while k > 0:
            lower, upper = kth, kth + 1  
            count = 0
            while lower <= n:  
                count += min(upper, n + 1) - lower
                lower *= 10
                upper *= 10
            if count <= k:  
                k -= count  
                kth += 1  
            else:
                k -= 1     
                kth *= 10  
        return kth
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def countArrangement(self, N):
        used = [False for _ in range(N + 1)]
        self.count = 0
        def helper(i):          
            if i == 0:          
                self.count += 1
                return
            for num in range(1, N + 1):
                if not used[num] and (num % i == 0 or i % num == 0):
                    used[num] = True
                    helper(i - 1)
                    used[num] = False
        helper(N)
        return self.count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution:
    def bagOfTokensScore(self, tokens, P):
        points, power = 0, P
        left, right = 0, len(tokens) - 1
        tokens.sort()
        while left < len(tokens) and tokens[left] <= power:
            power -= tokens[left]
            points += 1
            left += 1
        if not points:      
            return 0
        while right - left > 1:
            points -= 1
            power += tokens[right]
            right -= 1
            while right - left >= 0 and tokens[left] <= power:
                power -= tokens[left]
                points += 1
                left += 1
        return points
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def threeSumSmaller(self, nums, target):
        count = 0
        nums.sort()
        for i in range(len(nums)-2):
            left, right = i+1, len(nums)-1
            while left < right:
                if nums[i] + nums[left] + nums[right] < target:
                    count += right - left
                    left += 1
                else:
                    right -= 1
        return count
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def printTree(self, root):
        def height(node):
            if not node:
                return 0
            return 1 + max(height(node.left), height(node.right))
        rows = height(root)
        cols = 2 ** rows - 1
        result = [["" for _ in range(cols)] for _ in range(rows)]
        def place(node, r, c):
            if not node:
                return
            result[r][c] = str(node.val)
            shift = 2 ** (rows - r - 2)     
            place(node.left, r + 1, c - shift)
            place(node.right, r + 1, c + shift)
        place(root, 0, cols // 2)
        return result
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def magicalString(self, n):
        if n == 0:
            return 0
        i = 2  
        s = [1, 2, 2]
        ones = 1
        while len(s) < n:
            digit = s[-1] ^ 3  
            s.append(digit)
            if s[i] == 2:
                s.append(digit)
            if digit == 1:
                ones += s[i]
            i += 1
        if len(s) > n and s[-1] == 1:   
            ones -= 1
        return ones
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def deserialize(self, s):
        return self.helper(eval(s))  
    def helper(self, s_eval):
        if isinstance(s_eval, int):
            return NestedInteger(s_eval)
        nested = NestedInteger()  
        for item in s_eval:
            nested.add(self.helper(item))
        return nested
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def numSubarrayBoundedMax(self, A, L, R):
        subarrays, total = 0, 0 
        last_above_max = -1     
        for i, num in enumerate(A):
            if num > R:
                subarrays = 0
                last_above_max = i
            elif num < L:       
                total += subarrays
            else:               
                subarrays = i - last_above_max
                total += subarrays
        return total
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def peakIndexInMountainArray(self, nums):
        left, right = 1, len(nums) - 2
        while left < right:
            mid = (left + right) // 2
            if nums[mid + 1] < nums[mid]:
                right = mid
            else:
                left = mid + 1
        return left
EOF
class Solution(object):
    def isMajorityElement(self, nums, target):
        if len(nums) % 2:
            if nums[len(nums)//2] != target:
                return False
        else:
            if not (nums[len(nums)//2-1] == nums[len(nums)//2] == target):
                return False
        left = bisect.bisect_left(nums, target)
        right= bisect.bisect_right(nums, target)
        return (right-left)*2 > len(nums)
EOF
class Solution(object):
    def minimumAbsDifference(self, arr):
        result = []
        min_diff = float("inf")
        arr.sort()
        for i in xrange(len(arr)-1):
            diff = arr[i+1]-arr[i]
            if diff < min_diff:
                min_diff = diff
                result = [[arr[i], arr[i+1]]]
            elif diff == min_diff:
                result.append([arr[i], arr[i+1]])
        return result
EOF
class Solution(object):
    def countOfAtoms(self, formula):
        parse = re.findall(r"([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)", formula)
        stk = [collections.Counter()]
        for name, m1, left_open, right_open, m2 in parse:
            if name:
              stk[-1][name] += int(m1 or 1)
            if left_open:
              stk.append(collections.Counter())
            if right_open:
                top = stk.pop()
                for k, v in top.iteritems():
                  stk[-1][k] += v * int(m2 or 1)
        return "".join(name + (str(stk[-1][name]) if stk[-1][name] > 1 else '') \
                       for name in sorted(stk[-1]))
EOF
def angryProfessor(k, a):
    res = 'NO'
    if len(list(filter(lambda x: x <= 0, a))) < k:
        res = 'YES'
    return res
if __name__ == "__main__":
    t = int(input().strip())
    for a0 in range(t):
        n, k = input().strip().split(' ')
        n, k = [int(n), int(k)]
        a = list(map(int, input().strip().split(' ')))
        result = angryProfessor(k, a)
        print(result)
EOF
class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node
def print_singly_linked_list(node, sep, fptr):
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)
def reverse(head):
    cur = None
    while head:
        temp = head.next
        head.next = cur
        cur = head
        head = temp
    return cur
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        llist1 = reverse(llist.head)
        print_singly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()
EOF
for i in range(int(input())): 
    a = int(input()); A = set(input().split()) 
    b = int(input()); B = set(input().split())
    print(len((A - B)) == 0)
EOF
_author_ = 'jake'
_project_ = 'leetcode'
class Solution(object):
    def lengthOfLastWord(self, s):
        i = len(s) - 1
        end = -1
        while i >= 0:
            if s[i] == ' ' and end != -1:
                return end - i
            if s[i] != ' ' and end == -1:
                end = i
            i -= 1
        return end + 1 if end != -1 else 0
EOF
