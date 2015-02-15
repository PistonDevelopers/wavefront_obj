use std::cmp::Ordering;

/// Extensions to orderings.
pub trait OrderingExt {
  /// Lexicographically chains comparisions.
  fn lexico<F: Fn() -> Ordering>(self, f: F) -> Self;
}

impl OrderingExt for Ordering {
  fn lexico<F: Fn() -> Ordering>(self, f: F) -> Ordering {
    match self {
      Ordering::Less
    | Ordering::Greater => self,
      Ordering::Equal   => f(),
    }
  }
}
