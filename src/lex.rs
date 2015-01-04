use std::iter;
use std::str;

/// A parsing error, with location information.
#[derive(Show, PartialEq)]
pub struct ParseError {
  /// The line of input the error is on.
  pub line_number:   uint,
  /// The error message.
  pub message:       String,
}

#[inline]
fn is_whitespace(c: u8) -> bool {
  c == b' ' || c == b'\t' || c == b'\n'
}

pub struct Lexer<'a> {
  bytes: iter::Peekable<u8, str::Bytes<'a>>,
  current_line_number: uint,
}

impl<'a> Lexer<'a> {
  pub fn new(input: &'a str) -> Lexer<'a> {
    Lexer {
      bytes: input.bytes().peekable(),
      current_line_number: 1,
    }
  }

  /// Advance the lexer by one character.
  fn advance(&mut self) {
    match self.bytes.next() {
      None => {},
      Some(c) => {
        if c == b'\n' {
          self.current_line_number += 1;
        }
      }
    }
  }

  /// Looks at the next character the lexer is pointing to.
  fn peek(&mut self) -> Option<u8> {
    fn deref_u8(x: &u8) -> u8 { *x }
    self.bytes.peek().map(deref_u8)
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
      None => false,
      Some(c) => {
        if c == b'#' {
          // skip over the rest of the comment (except the newline)
          self.skip_unless(|c| c == b'\n');
          true
        } else {
          false
        }
      }
    }
  }

  fn skip_whitespace_except_newline(&mut self) -> bool {
    self.skip_while(|c| c == b'\t' || c == b' ')
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
          if c == b'#' {
            assert!(self.skip_comment());
            self.skip_whitespace_except_newline();
          } else if is_whitespace(c) {
            if c == b'\n' && ret.is_empty() {
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

impl<'a> Iterator for Lexer<'a> {
  type Item = String;

  fn next(&mut self) -> Option<String> {
    self.next_word().map(|buf| {
      match String::from_utf8(buf) {
        Ok(s) => s,
        Err(_) => panic!("Lex error: Invalid utf8 on line {}.", self.current_line_number),
      }
    })
  }

  fn size_hint(&self) -> (uint, Option<uint>) {
    (0, None)
  }
}

#[test]
fn test_next_word() {
  let mut l = Lexer::new("hello world\n this# is\na   \t test\n");
  assert_eq!(l.next_word(), Some(b"hello".to_vec()));
  assert_eq!(l.current_line_number, 1);
  assert_eq!(l.next_word(), Some(b"world".to_vec()));
  assert_eq!(l.current_line_number, 1);
  assert_eq!(l.next_word(), Some(b"\n".to_vec()));
  assert_eq!(l.current_line_number, 2);
  assert_eq!(l.next_word(), Some(b"this".to_vec()));
  assert_eq!(l.current_line_number, 2);
  assert_eq!(l.next_word(), Some(b"\n".to_vec()));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(b"a".to_vec()));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(b"test".to_vec()));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(b"\n".to_vec()));
  assert_eq!(l.current_line_number, 4);
  assert_eq!(l.next_word(), None);
}
