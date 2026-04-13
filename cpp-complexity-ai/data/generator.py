"""
Synthetic dataset generator.
Produces ~15,000 (C++ code, label) pairs from templates with random variation.
"""
import random
from typing import Tuple, List

def rand_var(): return random.choice(["arr", "nums", "vec", "data", "items"])
def rand_int_var(): return random.choice(["i", "j", "k", "idx"])
def rand_result(): return random.choice(["result", "sum", "total", "ans"])

# ── O(1) ────────────────────────────────────────────────────────────────────
def gen_o1() -> Tuple[str, str]:
    v = rand_var()
    t = random.choice([
        f"int get_first(vector<int>& {v}) {{ return {v}[0]; }}",
        f"bool is_empty(vector<int>& {v}) {{ return {v}.empty(); }}",
        "int add(int a, int b) { return a + b; }",
        "void swap_vars(int& a, int& b) { int tmp = a; a = b; b = tmp; }",
        "int max_of_two(int a, int b) { return (a > b) ? a : b; }",
    ])
    return t, "O(1)"

# ── O(log n) ────────────────────────────────────────────────────────────────
def gen_ologn() -> Tuple[str, str]:
    v = rand_var()
    t = random.choice([
        f"""int binary_search(vector<int>& {v}, int target) {{
    int left = 0, right = {v}.size() - 1;
    while (left <= right) {{
        int mid = left + (right - left) / 2;
        if ({v}[mid] == target) return mid;
        else if ({v}[mid] < target) left = mid + 1;
        else right = mid - 1;
    }}
    return -1;
}}""",
        """long long fast_pow(long long base, int exp) {
    long long result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) result *= base;
        base *= base; exp /= 2;
    }
    return result;
}""",
        """int count_bits(int n) {
    int count = 0;
    while (n > 0) { n >>= 1; count++; }
    return count;
}""",
    ])
    return t, "O(log n)"

# ── O(n) ────────────────────────────────────────────────────────────────────
def gen_on() -> Tuple[str, str]:
    v, i, r = rand_var(), rand_int_var(), rand_result()
    t = random.choice([
        f"""int linear_sum(vector<int>& {v}) {{
    int {r} = 0;
    for (int {i} = 0; {i} < {v}.size(); {i}++) {{ {r} += {v}[{i}]; }}
    return {r};
}}""",
        f"""int find_max(vector<int>& {v}) {{
    int {r} = {v}[0];
    for (int x : {v}) {{ if (x > {r}) {r} = x; }}
    return {r};
}}""",
        f"""bool has_duplicate(vector<int>& {v}) {{
    unordered_set<int> seen;
    for (int x : {v}) {{ if (seen.count(x)) return true; seen.insert(x); }}
    return false;
}}""",
        f"""void reverse_array(vector<int>& {v}) {{
    int left = 0, right = {v}.size() - 1;
    while (left < right) {{ swap({v}[left++], {v}[right--]); }}
}}""",
    ])
    return t, "O(n)"

# ── O(n log n) ──────────────────────────────────────────────────────────────
def gen_onlogn() -> Tuple[str, str]:
    v = rand_var()
    t = random.choice([
        f"""void merge_sort(vector<int>& {v}, int left, int right) {{
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    merge_sort({v}, left, mid);
    merge_sort({v}, mid + 1, right);
    vector<int> tmp;
    int i = left, j = mid + 1;
    while (i <= mid && j <= right) {{
        if ({v}[i] <= {v}[j]) tmp.push_back({v}[i++]);
        else tmp.push_back({v}[j++]);
    }}
    while (i <= mid) tmp.push_back({v}[i++]);
    while (j <= right) tmp.push_back({v}[j++]);
    for (int k = left; k <= right; k++) {v}[k] = tmp[k - left];
}}""",
        f"""vector<int> sort_and_dedup(vector<int> {v}) {{
    sort({v}.begin(), {v}.end());
    {v}.erase(unique({v}.begin(), {v}.end()), {v}.end());
    return {v};
}}""",
    ])
    return t, "O(n log n)"

# ── O(n^2) ──────────────────────────────────────────────────────────────────
def gen_on2() -> Tuple[str, str]:
    v, i = rand_var(), rand_int_var()
    t = random.choice([
        f"""void bubble_sort(vector<int>& {v}) {{
    int n = {v}.size();
    for (int {i} = 0; {i} < n - 1; {i}++)
        for (int j = 0; j < n - {i} - 1; j++)
            if ({v}[j] > {v}[j+1]) swap({v}[j], {v}[j+1]);
}}""",
        f"""bool has_pair_with_sum(vector<int>& {v}, int target) {{
    for (int {i} = 0; {i} < {v}.size(); {i}++)
        for (int j = {i}+1; j < {v}.size(); j++)
            if ({v}[{i}] + {v}[j] == target) return true;
    return false;
}}""",
        """vector<vector<int>> matrix_multiply(
    vector<vector<int>>& A, vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}""",
    ])
    return t, "O(n^2)"

# ── O(2^n) ──────────────────────────────────────────────────────────────────
def gen_o2n() -> Tuple[str, str]:
    t = random.choice([
        """int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}""",
        """int count_subsets(vector<int>& arr, int idx, int target) {
    if (target == 0) return 1;
    if (idx == (int)arr.size() || target < 0) return 0;
    return count_subsets(arr, idx + 1, target - arr[idx])
         + count_subsets(arr, idx + 1, target);
}""",
        """void generate_subsets(vector<int>& arr, int idx, vector<int> curr) {
    if (idx == (int)arr.size()) { return; }
    generate_subsets(arr, idx + 1, curr);
    curr.push_back(arr[idx]);
    generate_subsets(arr, idx + 1, curr);
}""",
    ])
    return t, "O(2^n)"

# ── O(n!) ───────────────────────────────────────────────────────────────────
def gen_onfactorial() -> Tuple[str, str]:
    t = random.choice([
        """void permutations(vector<int>& arr, int start) {
    if (start == (int)arr.size()) { return; }
    for (int i = start; i < (int)arr.size(); i++) {
        swap(arr[start], arr[i]);
        permutations(arr, start + 1);
        swap(arr[start], arr[i]);
    }
}""",
        """bool solve_tsp(vector<vector<int>>& dist, vector<bool>& visited,
               int curr, int count, int cost, int& ans) {
    if (count == (int)dist.size() && dist[curr][0]) {
        ans = min(ans, cost + dist[curr][0]);
        return true;
    }
    for (int i = 0; i < (int)dist.size(); i++) {
        if (!visited[i] && dist[curr][i]) {
            visited[i] = true;
            solve_tsp(dist, visited, i, count+1, cost+dist[curr][i], ans);
            visited[i] = false;
        }
    }
    return false;
}""",
    ])
    return t, "O(n!)"


# ── O(sqrt n) ────────────────────────────────────────────────────────────────
def gen_osqrtn() -> Tuple[str, str]:
    t = random.choice([
        """int count_divisors(int n) {
    int count = 0;
    for (int i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            count++;
            if (i != n / i) count++;
        }
    }
    return count;
}""",
        """bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}""",
        """int smallest_prime_factor(int n) {
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return i;
    }
    return n;
}""",
    ])
    return t, "O(sqrt n)"


# ── O(n sqrt n) ───────────────────────────────────────────────────────────────
def gen_onsqrtn() -> Tuple[str, str]:
    t = random.choice([
        """void sqrt_decomp_update(vector<int>& a, vector<int>& block, int l, int r, int val) {
    int B = sqrt(a.size());
    for (int i = l; i <= r; i++) {
        a[i] += val;
        block[i / B] += val;
    }
}""",
        """int sqrt_decomp_query(vector<int>& a, vector<int>& block, int l, int r) {
    int B = sqrt(a.size());
    int sum = 0;
    for (int i = l; i <= r; i++) {
        sum += a[i];
    }
    return sum;
}""",
        """vector<int> sieve_of_eratosthenes(int n) {
    vector<bool> is_prime(n + 1, true);
    vector<int> primes;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
            for (int j = i * i; j <= n; j += i++)
                is_prime[j] = false;
        }
    }
    return primes;
}""",
    ])
    return t, "O(n sqrt n)"


# ── O(n^2 log n) ──────────────────────────────────────────────────────────────
def gen_on2logn() -> Tuple[str, str]:
    t = random.choice([
        """void solve(vector<vector<int>>& a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sort(a[j].begin(), a[j].end());
        }
    }
}""",
        """int count_pairs_sorted(vector<int>& arr) {
    int n = arr.size(), count = 0;
    for (int i = 0; i < n; i++) {
        vector<int> tmp(arr.begin() + i, arr.end());
        sort(tmp.begin(), tmp.end());
        count += upper_bound(tmp.begin(), tmp.end(), arr[i]) - tmp.begin();
    }
    return count;
}""",
    ])
    return t, "O(n^2 log n)"


# ── O(n^3) ────────────────────────────────────────────────────────────────────
def gen_on3() -> Tuple[str, str]:
    t = random.choice([
        """void floyd_warshall(vector<vector<int>>& dist, int n) {
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
}""",
        """int count_triples(vector<int>& arr) {
    int n = arr.size(), count = 0;
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            for (int k = j+1; k < n; k++)
                if (arr[i] + arr[j] + arr[k] == 0) count++;
    return count;
}""",
        """vector<vector<int>> matrix_power(vector<vector<int>> A, int n) {
    int sz = A.size();
    vector<vector<int>> result(sz, vector<int>(sz, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < sz; j++)
            for (int k = 0; k < sz; k++)
                result[i][j] += A[i][k] * A[k][j];
    return result;
}""",
    ])
    return t, "O(n^3)"


# ── O(n + m) ──────────────────────────────────────────────────────────────────
def gen_onm_add() -> Tuple[str, str]:
    t = random.choice([
        """vector<int> bfs(vector<vector<int>>& adj, int start, int n) {
    vector<int> dist(n, -1);
    queue<int> q;
    q.push(start);
    dist[start] = 0;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int u : adj[v]) {
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                q.push(u);
            }
        }
    }
    return dist;
}""",
        """void dfs(vector<vector<int>>& adj, vector<bool>& visited, int v) {
    visited[v] = true;
    for (int u : adj[v]) {
        if (!visited[u]) dfs(adj, visited, u);
    }
}""",
        """vector<int> topological_sort(vector<vector<int>>& adj, int n) {
    vector<int> in_degree(n, 0), order;
    for (int v = 0; v < n; v++)
        for (int u : adj[v]) in_degree[u]++;
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (in_degree[i] == 0) q.push(i);
    while (!q.empty()) {
        int v = q.front(); q.pop();
        order.push_back(v);
        for (int u : adj[v])
            if (--in_degree[u] == 0) q.push(u);
    }
    return order;
}""",
    ])
    return t, "O(n + m)"


# ── O(n * m) ──────────────────────────────────────────────────────────────────
def gen_onm_mul() -> Tuple[str, str]:
    t = random.choice([
        """int lcs(string a, string b) {
    int n = a.size(), m = b.size();
    vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            dp[i][j] = (a[i-1]==b[j-1]) ? dp[i-1][j-1]+1
                                          : max(dp[i-1][j], dp[i][j-1]);
    return dp[n][m];
}""",
        """bool knapsack(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<bool>> dp(n+1, vector<bool>(W+1, false));
    dp[0][0] = true;
    for (int i = 1; i <= n; i++)
        for (int w = 0; w <= W; w++) {
            dp[i][w] = dp[i-1][w];
            if (w >= weights[i-1])
                dp[i][w] = dp[i][w] || dp[i-1][w-weights[i-1]];
        }
    return dp[n][W];
}""",
        """vector<vector<int>> grid_dp(vector<vector<int>>& grid, int n, int m) {
    vector<vector<int>> dp(n, vector<int>(m, 0));
    dp[0][0] = grid[0][0];
    for (int i = 1; i < n; i++) dp[i][0] = dp[i-1][0] + grid[i][0];
    for (int j = 1; j < m; j++) dp[0][j] = dp[0][j-1] + grid[0][j];
    for (int i = 1; i < n; i++)
        for (int j = 1; j < m; j++)
            dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j];
    return dp;
}""",
    ])
    return t, "O(n * m)"


# ── O((n + q) * sqrt(n)) ─────────────────────────────────────────────────────
def gen_onq_sqrtn() -> Tuple[str, str]:
    t = random.choice([
        """void process_queries(vector<int>& a, int n, int q) {
    int B = (int)sqrt(n);
    vector<int> block(n / B + 1, 0);
    // init blocks
    for (int i = 0; i < n; i++) block[i / B] += a[i];
    // handle q queries each O(sqrt n)
    for (int query = 0; query < q; query++) {
        int l, r, val; cin >> l >> r >> val;
        for (int i = l; i <= min(r, (l/B+1)*B-1); i++) {
            a[i] += val; block[i/B] += val;
        }
        for (int b = l/B+1; b < r/B; b++) block[b] += val * B;
        for (int i = max(l, (r/B)*B); i <= r; i++) {
            a[i] += val; block[i/B] += val;
        }
    }
}""",
        """struct SqrtDecomp {
    int B;
    vector<int> a, block, lazy;
    SqrtDecomp(vector<int>& arr) {
        int n = arr.size();
        B = sqrt(n);
        a = arr;
        block.assign(n/B+1, 0);
        lazy.assign(n/B+1, 0);
        for (int i = 0; i < n; i++) block[i/B] += a[i];
    }
    void update(int l, int r, int val) {
        for (int i = l; i <= r && i < (int)a.size(); i++) {
            a[i] += val; block[i/B] += val;
        }
    }
    int query(int l, int r) {
        int sum = 0;
        for (int i = l; i <= r && i < (int)a.size(); i++) sum += a[i];
        return sum;
    }
};""",
    ])
    return t, "O((n + q) * sqrt(n))"


COMPLEXITY_CLASSES = [
    "O(1)",
    "O(log n)",
    "O(sqrt n)",
    "O(n)",
    "O(n log n)",
    "O(n sqrt n)",
    "O(n^2)",
    "O(n^2 log n)",
    "O(n^3)",
    "O(n + m)",
    "O(n * m)",
    "O((n + q) * sqrt(n))",
    "O(2^n)",
    "O(n!)",
]

GENERATORS = {
    "O(1)":                  (gen_o1,           2000),
    "O(log n)":              (gen_ologn,         1500),
    "O(sqrt n)":             (gen_osqrtn,        1000),
    "O(n)":                  (gen_on,            2000),
    "O(n log n)":            (gen_onlogn,        2000),
    "O(n sqrt n)":           (gen_onsqrtn,       1000),
    "O(n^2)":                (gen_on2,           2000),
    "O(n^2 log n)":          (gen_on2logn,        800),
    "O(n^3)":                (gen_on3,            800),
    "O(n + m)":              (gen_onm_add,       1500),
    "O(n * m)":              (gen_onm_mul,       1500),
    "O((n + q) * sqrt(n))":  (gen_onq_sqrtn,     800),
    "O(2^n)":                (gen_o2n,           1000),
    "O(n!)":                 (gen_onfactorial,    500),
}


def build_dataset(seed: int = 42):
    from parser.ast_parser import COMPLEXITY_CLASSES as _C
    random.seed(seed)
    dataset = []
    for label, (gen_fn, count) in GENERATORS.items():
        for _ in range(count):
            code, lbl = gen_fn()
            dataset.append((code.strip(), lbl))
    random.shuffle(dataset)
    print(f"Dataset size: {len(dataset)}")
    for lbl in _C:
        n = sum(1 for _, l in dataset if l == lbl)
        print(f"  {lbl}: {n}")
    return dataset