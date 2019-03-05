use std::str;

/// A parsing error, with location information.
#[derive(Debug, PartialEq)]
pub struct ParseError {
  /// The line of input the error is on.
  pub line_number: usize,
  /// The error message.
  pub message: String,
}

#[inline]
fn is_whitespace_except_newline(c: u8) -> bool {
  c == b' ' || c == b'\t' || c == b'\r'
}

#[inline]
fn is_whitespace(c: u8) -> bool {
  is_whitespace_except_newline(c) || c == b'\n'
}

#[derive(Clone)]
pub struct Lexer<'a> {
  bytes: &'a [u8],
  read_pos: usize,
  current_line_number: usize,
}

impl<'a> Lexer<'a> {
  pub fn new(input: &'a str) -> Lexer<'a> {
    Lexer {
      bytes: input.as_bytes(),
      read_pos: 0,
      current_line_number: 1,
    }
  }

  /// Advance the lexer by one character.
  fn advance(&mut self) {
    match self.peek() {
      Some(&c) if c == b'\n' => {
        self.current_line_number += 1;
      }
      _ => {}
    }
    self.read_pos += 1;
  }

  /// Looks at the next character the lexer is pointing to.
  fn peek(&self) -> Option<&u8> {
    self.bytes.get(self.read_pos)
  }

  /// Advance past characters until the given condition is true.
  ///
  /// Returns whether or not any of the input was skipped.
  ///
  /// Postcondition: Either the end of the input was reached or
  /// `is_true` returns false for the currently peekable character.
  fn skip_while<F: Fn(u8) -> bool>(&mut self, is_true: F) -> bool {
    let mut was_anything_skipped = false;

    loop {
      match self.peek() {
        None => break,
        Some(&c) if !is_true(c) => break,
        _ => {
          self.advance();
          was_anything_skipped = true;
        }
      }
    }

    debug_assert!(self.peek().map(|&c| !is_true(c)).unwrap_or(true));

    was_anything_skipped
  }

  /// Advance past characters until the given condition is true.
  ///
  /// Returns whether or not any of the input was skipped.
  ///
  /// Postcondition: Either the end of the input was reached or
  /// `is_false` returns true for the currently peekable character.
  fn skip_unless<F: Fn(u8) -> bool>(&mut self, is_false: F) -> bool {
    self.skip_while(|c| !is_false(c))
  }

  /// Advances past comments in the input, including their trailing newlines.
  ///
  /// Returns whether or not any of the input was skipped.
  fn skip_comment(&mut self) -> bool {
    match self.peek() {
      Some(b'#') => {
        // skip over the rest of the comment (except the newline)
        self.skip_unless(|c| c == b'\n');
        true
      }
      _ => false,
    }
  }

  fn skip_whitespace_except_newline(&mut self) -> bool {
    self.skip_while(is_whitespace_except_newline)
  }

  /// Gets the next word in the input, as well as whether it's on
  /// a different line than the last word we got.
  fn next_word(&mut self) -> Option<&'a [u8]> {
    self.skip_whitespace_except_newline();
    self.skip_comment();

    let start_ptr = self.read_pos;

    match self.peek() {
      Some(b'\n') => {
        self.advance();
        Some(&self.bytes[start_ptr..self.read_pos]) // newline
      }
      Some(_) => {
        if self.skip_unless(|c| is_whitespace(c) || c == b'#') {
          Some(&self.bytes[start_ptr..self.read_pos])
        } else {
          None
        }
      }
      None => None,
    }
  }
}

#[derive(Clone)]
pub struct PeekableLexer<'a> {
  inner: Lexer<'a>,
  peeked: Option<Option<&'a str>>,
}

impl<'a> PeekableLexer<'a> {
  pub fn new(lexer: Lexer<'a>) -> Self {
    Self {
      inner: lexer,
      peeked: None,
    }
  }
  pub fn next_str(&mut self) -> Option<&'a str> {
    match self.peeked.take() {
      Some(v) => v,
      None => self
        .inner
        .next_word()
        .map(|buf| unsafe { str::from_utf8_unchecked(buf) }),
    }
  }

  pub fn peek_str(&mut self) -> Option<&'a str> {
    match self.peeked {
      Some(v) => v,
      None => {
        let peek = self
          .inner
          .next_word()
          .map(|buf| unsafe { str::from_utf8_unchecked(buf) });

        self.peeked.replace(peek);
        peek
      }
    }
  }
}

#[test]
fn test_next_word() {
  let mut l = Lexer::new("hello wor\rld\n this# is\r\na   \t test\n");
  assert_eq!(l.next_word(), Some(&b"hello"[..]));
  assert_eq!(l.current_line_number, 1);
  assert_eq!(l.next_word(), Some(&b"wor"[..]));
  assert_eq!(l.current_line_number, 1);
  assert_eq!(l.next_word(), Some(&b"ld"[..]));
  assert_eq!(l.current_line_number, 1);
  assert_eq!(l.next_word(), Some(&b"\n"[..]));
  assert_eq!(l.current_line_number, 2);
  assert_eq!(l.next_word(), Some(&b"this"[..]));
  assert_eq!(l.current_line_number, 2);
  assert_eq!(l.next_word(), Some(&b"\n"[..]));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(&b"a"[..]));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(&b"test"[..]));
  assert_eq!(l.current_line_number, 3);
  assert_eq!(l.next_word(), Some(&b"\n"[..]));
  assert_eq!(l.current_line_number, 4);
  assert_eq!(l.next_word(), None);
}
