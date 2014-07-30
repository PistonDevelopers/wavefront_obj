use std::iter;
use std::mem;
use std::slice;

/// A parsing error, with location information.
#[deriving(Show, PartialEq)]
pub struct ParseError {
  /// The line of input the error is on.
  pub line_number:   uint,
  /// The error message.
  pub message:       String,
}

#[inline]
fn c2u8(c: char) -> u8 {
  c as u8
}

#[allow(dead_code)]
fn s2u8(s: &str) -> Vec<u8> {
  s.bytes().collect()
}

#[allow(dead_code)]
fn u82s<'a>(u8s: &'a [u8]) -> &'a str {
  // According to the docs, &'a str is the same representation as a &'a [u8].
  unsafe { mem::transmute(u8s) }
}

#[test]
fn verify_u82s() {
  let cases = [ "hello", "world" ];

  for &case in cases.iter() {
    assert_eq!(u82s(s2u8(case).as_slice()), case);
  }
}

#[inline]
fn is_whitespace(c: u8) -> bool {
  c == c2u8(' ') || c == c2u8('\t') || c == c2u8('\n')
}

pub struct Lexer<'a> {
  bytes: iter::Peekable<u8, iter::Fuse<iter::Map<'a, &'a u8, u8, slice::Items<'a, u8>>>>,
  current_line_number: uint,
}

impl<'a> Lexer<'a> {
  pub fn new(input: &'a str) -> Lexer<'a> {
    Lexer {
      bytes: input.bytes().fuse().peekable(),
      current_line_number: 1,
    }
  }

  /// Advance the lexer by one character.
  fn advance(&mut self) {
    match self.bytes.next() {
      None => {},
      Some(c) => {
        if c == c2u8('\n') {
          self.current_line_number += 1;
        }
      }
    }
  }

  /// Looks at the next character the lexer is pointing to.
  fn peek(&mut self) -> Option<u8> {
    self.bytes.peek().map(|c| *c)
  }

  /// Advance past characters until the given condition is true.
  ///
  /// Returns whether or not any of the input was skipped.
  ///
  /// Postcondition: Either the end of the input was reached or
  /// `is_true` returns false for the currently peekable character.
  fn skip_while(&mut self, is_true: |u8| -> bool) -> bool {
    let mut was_anything_skipped = false;

    loop {
      match self.peek() {
        None => break,
        Some(c) => {
          if !is_true(c) { break }
        }
      }
      self.advance();
      was_anything_skipped = true;
    }

    debug_assert!(self.peek().map(|c| !is_true(c)).unwrap_or(true));

    was_anything_skipped
  }

  /// Advance past characters until the given condition is true.
  ///
  /// Returns whether or not any of the input was skipped.
  ///
  /// Postcondition: Either the end of the input was reached or
  /// `is_false` returns true for the currently peekable character.
  fn skip_unless(&mut self, is_false: |u8| -> bool) -> bool {
    self.skip_while(|c| !is_false(c))
  }

  /// Advances past comments in the input, including their trailing newlines.
  ///
  /// Returns whether or not any of the input was skipped.
  fn skip_comment(&mut self) -> bool {
    match self.peek() {
      None => return false,
      Some(c) => {
        if c == c2u8('#') {
          // skip over the rest of the comment (except the newline)
          self.skip_unless(|c| c == c2u8('\n'));
          return true;
        } else {
          return false;
        }
      }
    }
  }

  fn skip_whitespace_except_newline(&mut self) -> bool {
    self.skip_while(|c| c == c2u8('\t') || c == c2u8(' '))
  }

  /// Gets the next word in the input, as well as whether it's on
  /// a different line than the last word we got.
  fn next_word(&mut self) -> Option<Vec<u8>> {
    let mut ret: Vec<u8> = Vec::new();

    self.skip_whitespace_except_newline();

    loop {
      match self.peek() {
        None => break,
        Some(c) => {
          if c == c2u8('#') {
            assert!(self.skip_comment());
            self.skip_whitespace_except_newline();
          } else if is_whitespace(c) {
            if c == c2u8('\n') && ret.is_empty() {
              ret.push(c);
              self.advance();
            }
            break;
          } else {
            ret.push(c);
            self.advance();
          }
        }
      }
    }

    if ret.is_empty() {
      debug_assert_eq!(self.peek(), None);
      None
    } else {
      Some(ret)
    }
  }
}

impl<'a> Iterator<String> for Lexer<'a> {
  fn next(&mut self) -> Option<String> {
    match self.next_word() {
      None => None,
      Some(buf) => {
        match String::from_utf8(buf) {
          Ok(s) => Some(s),
          Err(_) => fail!("Lex error: Invalid utf8 on line {}.", self.current_line_number),
        }
      }
    }
  }

  fn size_hint(&self) -> (uint, Option<uint>) {
    (0, None)
  }
}

#[test]
fn test_next_word() {
  let mut l = Lexer::new("hello world\n this# is\na   \t test\n");
  assert_eq!(l.next_word(), Some(s2u8("hello")));
  assert_eq!(l.current_line_number, 1);
  assert_eq!(l.next_word(), Some(s2u8("world")));
  assert_eq!(l.current_line_number, 1);
  assert_eq!(l.next_word(), Some(s2u8("\n")));
  assert_eq!(l.current_line_number, 2);
  assert_eq!(l.next_word(), Some(s2u8("this")));
  assert_eq!(l.current_line_number, 2);
  assert_eq!(l.next_word(), Some(s2u8("\n")));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(s2u8("a")));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(s2u8("test")));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(s2u8("\n")));
  assert_eq!(l.current_line_number, 4);
  assert_eq!(l.next_word(), None);
}

#[test]
fn test_simd() {
  use std::simd;

  let x = simd::f32x4(1.0,2.0,3.0,4.0);

  let simd::f32x4(a, b, c, d) = x;

  assert_eq!(a, 1.0);
  assert_eq!(b, 2.0);
  assert_eq!(c, 3.0);
  assert_eq!(d, 4.0);
}
