pub fn public_fn(input: &str) -> String {
    let trimmed = input.trim();
    let mut out = String::new();
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            continue;
        }
        out.push(ch);
    }
    out
}

fn private_helper(a: u32, b: u32) -> u32 {
    let sum = a + b;
    match sum {
        0 => 0,
        n if n > 100 => 100,
        _ => sum,
    }
}

pub struct Counter {
    count: u64,
}

impl Counter {
    pub fn new() -> Self {
        Counter { count: 0 }
    }

    pub fn increment(&mut self) -> u64 {
        self.count += 1;
        self.count
    }
}
