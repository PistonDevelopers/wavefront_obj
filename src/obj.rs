//! A parser for Wavefront's `.obj` file format.
#![crate_type = "lib"]
#![deny(warnings)]
#![deny(missing_doc)]

use std::iter;
use std::mem;
use std::result;
use std::result::{Result,Ok,Err};
use std::slice;

/// A set of objects, as listed in an `.obj` file.
#[deriving(Clone, Show, PartialEq)]
pub struct ObjSet {
  /// Which material library to use.
  pub material_library: String,
  /// The set of objects.
  pub objects: Vec<Object>,
}

/// A mesh object.
#[deriving(Clone, Show, PartialEq)]
pub struct Object {
  /// A human-readable name for this object. This can be set in blender.
  pub name: String,
  /// The set of verticies this object is composed of. These are referenced
  /// by index in `faces`.
  pub verticies: Vec<Vertex>,
  /// A set of shapes (with materials applied to them) of which this object is
  // composed.
  pub geometry: Vec<Geometry>,
}

/// A set of shapes, all using the given material.
#[deriving(Clone, Show, PartialEq)]
pub struct Geometry {
  /// A reference to the material to apply to this geometry.
  pub material_name: Option<String>,
  /// Should we use smooth shading when rendering this?
  pub use_smooth_shading: bool,
  /// The shapes of which this geometry is composed.
  pub shapes: Vec<Shape>,
}

/// The various shapes supported by this library.
///
/// Convex polygons more complicated than a triangle are automatically
/// converted into triangles.
#[deriving(Clone, Show, Hash, PartialEq)]
pub enum Shape {
  /// A point specified by its position.
  Point(VertexIndex),
  /// A line specified by its endpoints.
  Line(VertexIndex, VertexIndex),
  /// A triangle specified by its three verticies.
  Triangle(VertexIndex, VertexIndex, VertexIndex),
}

/// A single 3-dimensional point on the corner of an object.
#[allow(missing_doc)]
#[deriving(Clone, Copy, Show)]
pub struct Vertex {
  pub x: f64,
  pub y: f64,
  pub z: f64,
}

fn fuzzy_cmp(x: f64, y: f64, delta: f64) -> Ordering {
  if (x - y).abs() <= delta {
    Equal
  } else if x < y {
    Greater
  } else {
    Less
  }
}

// TODO(cgaebel): Can we implement Eq here?
impl PartialEq for Vertex {
  fn eq(&self, other: &Vertex) -> bool {
    self.partial_cmp(other).unwrap() == Equal
  }
}

impl PartialOrd for Vertex {
  fn partial_cmp(&self, other: &Vertex) -> Option<Ordering> {
    Some(fuzzy_cmp(self.x, other.x, 0.00001)
      .cmp(&fuzzy_cmp(self.y, other.y, 0.00001))
      .cmp(&fuzzy_cmp(self.z, other.z, 0.00001)))
  }
}

/// An index into the `verticies` array of an object, representing a vertex in
/// the mesh. After parsing, this is guaranteed to be a valid index into the
/// array, so unchecked indexing may be used.
pub type VertexIndex = uint;

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

struct Lexer<'a> {
  bytes: iter::Peekable<u8, iter::Fuse<iter::Map<'a, &'a u8, u8, slice::Items<'a, u8>>>>,
  current_line_number: uint,
}

impl<'a> Lexer<'a> {
  fn new(input: &'a str) -> Lexer<'a> {
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

/// Slices the underlying string in an option.
fn sliced<'a>(s: &'a Option<String>) -> Option<&'a str> {
  match *s {
    None => None,
    Some(ref s) => Some(s.as_slice()),
  }
}

/// Blender exports shapes as a list of the verticies representing their corner.
/// This function turns that into a set of OpenGL-usable shapes.
fn to_triangles(xs: &[VertexIndex]) -> Vec<Shape> {
  match xs.len() {
    0 => return vec!(),
    1 => return vec!(Point(xs[0])),
    2 => return vec!(Line(xs[0], xs[1])),
    _ => {},
  }

  let last_elem = *xs.last().unwrap();

  xs.slice_to(xs.len()-1)
    .iter()
    .zip(xs.slice(1, xs.len()-1).iter())
    .map(|(&x, &y)| Triangle(last_elem, x, y))
    .collect()
}

#[test]
fn test_to_triangles() {
  assert_eq!(to_triangles(&[]), vec!());
  assert_eq!(to_triangles(&[3]), vec!(Point(3)));
  assert_eq!(to_triangles(&[1,2]), vec!(Line(1,2)));
  assert_eq!(to_triangles(&[1,2,3]), vec!(Triangle(3,1,2)));
  assert_eq!(to_triangles(&[1,2,3,4]), vec!(Triangle(4,1,2),Triangle(4,2,3)));
  assert_eq!(to_triangles(&[1,2,3,4,5]), vec!(Triangle(5,1,2),Triangle(5,2,3),Triangle(5,3,4)));
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

struct Parser<'a> {
  line_number: uint,
  lexer: iter::Peekable<String, iter::Fuse<Lexer<'a>>>,
}

impl<'a> Parser<'a> {
  fn new<'a>(input: &'a str) -> Parser<'a> {
    Parser {
      line_number: 1,
      lexer: Lexer::new(input).fuse().peekable(),
    }
  }

  fn error<A>(&self, msg: String) -> Result<A, ParseError> {
    result::Err(ParseError {
      line_number: self.line_number,
      message:     msg,
    })
  }

  fn next(&mut self) -> Option<String> {
    // TODO(cgaebel): This has a lot of useless allocations. Techincally we can
    // just be using slices into the underlying buffer instead of allocating a
    // new string for every single token. Unfortunately, I'm not sure how to
    // structure this to appease the borrow checker.
    let ret = self.lexer.next().map(|s| s.clone());

    match ret {
      None => {},
      Some(ref s) =>
        if s.as_slice() == "\n" {
          self.line_number += 1;
        },
    }

    ret
  }

  fn advance(&mut self) {
    self.next();
  }

  fn peek(&mut self) -> Option<String> {
    // TODO(cgaebel): See the comment in `next`.
    self.lexer.peek().map(|s| s.clone())
  }

  /// Possibly skips over some newlines.
  fn zero_or_more_newlines(&mut self) {
    loop {
      match sliced(&self.peek()) {
        None       => break,
        Some("\n") => {},
        Some(_)    => break,
      }
      self.advance();
    }
  }

  /// Skips over some newlines, failing if it didn't manage to skip any.
  fn one_or_more_newlines(&mut self) -> Result<(), ParseError> {
    match sliced(&self.peek()) {
      None => return self.error("Expected newline but got end of input.".into_string()),
      Some("\n") => {},
      Some(s) => return self.error(format!("Expected newline but got {}", s)),
    }

    self.zero_or_more_newlines();

    Ok(())
  }

  fn parse_material_library<'a>(&mut self) -> Result<String, ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error("Expected `mtllib` but got end of input.".into_string()),
      Some("mtllib") =>
        {},
      Some(got) =>
        return self.error(format!("Expected `mtllib` but got {}.", got)),
    }

    match self.next() {
      None =>
        self.error("Expected library name but got end of input.".into_string()),
      Some(got) =>
        Ok(got),
    }
  }

  fn parse_object_name(&mut self) -> Result<String, ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error(format!("Expected `o` but got end of input.")),
      Some("o") =>
        {},
      Some(got) =>
        return self.error(format!("Expected `o` but got {}.", got)),
    }

    match self.next() {
      None =>
        return self.error("Expected object name but got end of input.".into_string()),
      Some(got) =>
        Ok(got),
    }
  }

  // TODO(cgaebel): Should this be returning `num::rational::BigRational` instead?
  // I can't think of a good reason to do this except to make testing easier.
  fn parse_double(&mut self) -> Result<f64, ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error("Expected f64 but got end of input.".into_string()),
      Some(s) => {
        match from_str::<f64>(s) {
          None =>
            return self.error(format!("Expected f64 but got {}.", s)),
          Some(ret) =>
            Ok(ret)
        }
      }
    }
  }

  fn parse_vertex(&mut self) -> Result<Vertex, ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error("Expected `v` but got end of input.".into_string()),
      Some("v") =>
        {},
      Some(s) =>
        return self.error(format!("Expected `v` but got {}.", s)),
    }

    let x = try!(self.parse_double());
    let y = try!(self.parse_double());
    let z = try!(self.parse_double());

    Ok(Vertex { x: x, y: y, z: z })
  }

  /// BUG: Also munches trailing whitespace.
  fn parse_verticies(&mut self) -> Result<Vec<Vertex>, ParseError> {
    let mut result = Vec::new();

    loop {
      match sliced(&self.peek()) {
        Some("v") => {
          let v = try!(self.parse_vertex());
          result.push(v);
        },
        _ => break,
      }

      try!(self.one_or_more_newlines());
    }

    Ok(result)
  }

  fn parse_usemtl(&mut self) -> Result<String, ParseError> {
    match sliced(&self.next()) {
      Some("usemtl") => {},
      None    => return self.error("Expected `usemtl` but got end of input.".into_string()),
      Some(s) => return self.error(format!("Expected `usemtl` but got {}.", s)),
    }

    match self.next() {
      None    => return self.error("Expected material name but got end of input.".into_string()),
      Some(s) => Ok(s)
    }
  }

  fn parse_smooth_shading(&mut self) -> Result<bool, ParseError> {
    match sliced(&self.next()) {
      Some("s") => {},
      None      => return self.error("Expected `s` but got end of input.".into_string()),
      Some(s)   => return self.error(format!("Expected `s` but got {}.", s)),
    }

    match sliced(&self.next()) {
      Some("on")  => Ok(true),
      Some("off") => Ok(false),
      None        => return self.error("Expected `on` or `off` but got end of input.".into_string()),
      Some(s)     => return self.error(format!("Expected `on` or `off` but got {}.", s)),
    }
  }

  fn parse_int(&mut self) -> Result<int, ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error("Expected int but got end of input.".into_string()),
      Some(s) => {
        match from_str::<int>(s) {
          None =>
            return self.error(format!("Expected int but got {}.", s)),
          Some(ret) =>
            Ok(ret)
        }
      }
    }
  }

  /// `valid_verticies` is a range of valid verticies, in the range [min, max).
  fn parse_vertex_index(&mut self, valid_verticies: (uint, uint)) -> Result<VertexIndex, ParseError> {
    let mut x : int = try!(self.parse_int());

    let (min, max) = valid_verticies;

    // Handle negative vertex indexes.
    if x < 0 {
      x = max as int - x;
    }

    if x >= min as int && x < max as int {
      assert!(x > 0);
      Ok((x - min as int) as uint)
    } else {
      self.error(format!("Expected vertex in the range [{}, {}), but got {}.", min, max, x))
    }
  }

  fn parse_face(&mut self, valid_verticies: (uint, uint)) -> Result<Vec<Shape>, ParseError> {
    match sliced(&self.next()) {
      Some("f") => {},
      Some("l") => {},
      None      => return self.error("Expected `f` or `l` but got end of input.".into_string()),
      Some(s)   => return self.error(format!("Expected `f` or `l` but got {}.", s)),
    }

    let mut corner_list = Vec::new();

    corner_list.push(try!(self.parse_vertex_index(valid_verticies)));

    loop {
      match sliced(&self.peek()) {
        None       => break,
        Some("\n") => break,
        Some( _  ) => corner_list.push(try!(self.parse_vertex_index(valid_verticies))),
      }
    }

    Ok(to_triangles(corner_list.as_slice()))
  }

  fn parse_geometries(&mut self, valid_verticies: (uint, uint)) -> Result<Vec<Geometry>, ParseError> {
    let mut result = Vec::new();
    let mut shapes = Vec::new();

    let mut current_material   = None;
    let mut use_smooth_shading = false;

    loop {
      match sliced(&self.peek()) {
        Some("usemtl") => {
          let old_material =
            mem::replace(
              &mut current_material,
              Some(try!(self.parse_usemtl())));

          result.push(Geometry {
            material_name:      old_material,
            use_smooth_shading: use_smooth_shading,
            shapes:             mem::replace(&mut shapes, Vec::new()),
          });
        },
        Some("s") => {
          let old_smooth_shading =
            mem::replace(
              &mut use_smooth_shading,
              try!(self.parse_smooth_shading()));

          result.push(Geometry {
            material_name:      current_material.clone(),
            use_smooth_shading: old_smooth_shading,
            shapes:             mem::replace(&mut shapes, Vec::new()),
          })
        },
        Some("f") | Some("l") => {
          shapes.push_all(try!(self.parse_face(valid_verticies)).as_slice());
        },
        _ => break,
      }

      try!(self.one_or_more_newlines());
    }

    result.push(Geometry {
      material_name:      current_material,
      use_smooth_shading: use_smooth_shading,
      shapes:             shapes,
    });

    Ok(result.move_iter().filter(|ref x| !x.shapes.is_empty()).collect())
  }

  fn parse_object(&mut self, min_vertex_index: &mut uint, max_vertex_index: &mut uint) -> Result<Object, ParseError> {
    let name      = try!(self.parse_object_name());
    try!(self.one_or_more_newlines());
    let verticies = try!(self.parse_verticies());
    *max_vertex_index += verticies.len();
    let geometry  = try!(self.parse_geometries((*min_vertex_index, *max_vertex_index)));
    *min_vertex_index += verticies.len();

    Ok(Object {
      name:      name,
      verticies: verticies,
      geometry:  geometry,
    })
  }

  fn parse_objects(&mut self) -> Result<Vec<Object>, ParseError> {
    let mut result = Vec::new();

    let mut min_vertex_index = 1;
    let mut max_vertex_index = 1;

    loop {
      match sliced(&self.peek()) {
        Some("o") => result.push(try!(self.parse_object(&mut min_vertex_index, &mut max_vertex_index))),
        _         => break,
      }
    }

    Ok(result)
  }

  fn parse_objset(&mut self) -> Result<ObjSet, ParseError> {
    self.zero_or_more_newlines();

    let material_library = try!(self.parse_material_library());
    try!(self.one_or_more_newlines());
    let objects          = try!(self.parse_objects());

    self.zero_or_more_newlines();

    match self.peek() {
      None =>
        {},
      Some(s) =>
        return self.error(format!("Expected end of input but got {}.", s)),
    }

    Ok(ObjSet {
      material_library: material_library,
      objects:          objects,
    })
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
fn test_parse() {
  let test_case =
r#"
# Blender v2.69 (sub 0) OBJ File: ''
# www.blender.org
mtllib untitled.mtl
o Cube.001
v -1.000000 -1.000000 1.000000
v -1.000000 -1.000000 -1.000000
v 1.000000 -1.000000 -1.000000
v 1.000000 -1.000000 1.000000
v -1.000000 1.000000 1.000000
v -1.000000 1.000000 -1.000000
v 1.000000 1.000000 -1.000000
v 1.000000 1.000000 1.000000
usemtl None
s off
f 5 6 2 1
f 6 7 3 2
f 7 8 4 3
f 8 5 1 4
f 1 2 3 4
f 8 7 6 5
o Circle
v 0.000000 0.000000 -1.000000
v -0.195090 0.000000 -0.980785
v -0.382683 0.000000 -0.923880
v -0.555570 0.000000 -0.831470
v -0.707107 0.000000 -0.707107
v -0.831470 0.000000 -0.555570
v -0.923880 0.000000 -0.382683
v -0.980785 0.000000 -0.195090
v -1.000000 0.000000 -0.000000
v -0.980785 0.000000 0.195090
v -0.923880 0.000000 0.382683
v -0.831470 0.000000 0.555570
v -0.707107 0.000000 0.707107
v -0.555570 0.000000 0.831470
v -0.382683 0.000000 0.923880
v -0.195090 0.000000 0.980785
v 0.000000 0.000000 1.000000
v 0.195091 0.000000 0.980785
v 0.382684 0.000000 0.923879
v 0.555571 0.000000 0.831469
v 0.707107 0.000000 0.707106
v 0.831470 0.000000 0.555570
v 0.923880 0.000000 0.382683
v 0.980785 0.000000 0.195089
v 1.000000 0.000000 -0.000001
v 0.980785 0.000000 -0.195091
v 0.923879 0.000000 -0.382684
v 0.831469 0.000000 -0.555571
v 0.707106 0.000000 -0.707108
v 0.555569 0.000000 -0.831470
v 0.382682 0.000000 -0.923880
v 0.195089 0.000000 -0.980786
l 10 9
l 11 10
l 12 11
l 13 12
l 14 13
l 15 14
l 16 15
l 17 16
l 18 17
l 19 18
l 20 19
l 21 20
l 22 21
l 23 22
l 24 23
l 25 24
l 26 25
l 27 26
l 28 27
l 29 28
l 30 29
l 31 30
l 32 31
l 33 32
l 34 33
l 35 34
l 36 35
l 37 36
l 38 37
l 39 38
l 40 39
l 9 40
o Cube
v 1.000000 -1.000000 -1.000000
v 1.000000 -1.000000 1.000000
v -1.000000 -1.000000 1.000000
v -1.000000 -1.000000 -1.000000
v 1.000000 1.000000 -0.999999
v 0.999999 1.000000 1.000001
v -1.000000 1.000000 1.000000
v -1.000000 1.000000 -1.000000
usemtl Material
s off
f 41 42 43 44
f 45 48 47 46
f 41 45 46 42
f 42 46 47 43
f 43 47 48 44
f 45 41 44 48
"#;

  let expected =
     Ok(ObjSet {
      material_library: "untitled.mtl".into_string(),
      objects: vec!(
        Object {
          name: "Cube.001".into_string(),
          verticies: vec!(
            Vertex { x: -1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: 1.0, z: 1.0 },
            Vertex { x: -1.0, y: 1.0, z: -1.0 },
            Vertex { x: 1.0, y: 1.0, z: -1.0 },
            Vertex { x: 1.0, y: 1.0, z: 1.0 }
          ),
          geometry: vec!(
            Geometry {
              material_name: Some("None".into_string()),
              use_smooth_shading: false,
              shapes: vec!(
                Triangle(0, 4, 5),
                Triangle(0, 5, 1),
                Triangle(1, 5, 6),
                Triangle(1, 6, 2),
                Triangle(2, 6, 7),
                Triangle(2, 7, 3),
                Triangle(3, 7, 4),
                Triangle(3, 4, 0),
                Triangle(3, 0, 1),
                Triangle(3, 1, 2),
                Triangle(4, 7, 6),
                Triangle(4, 6, 5),
              )
            }
          )
        },
        Object {
          name: "Circle".into_string(),
          verticies: vec!(
            Vertex { x: 0.0, y: 0.0, z: -1.0 },
            Vertex { x: -0.19509, y: 0.0, z: -0.980785 },
            Vertex { x: -0.382683, y: 0.0, z: -0.92388 },
            Vertex { x: -0.55557, y: 0.0, z: -0.83147 },
            Vertex { x: -0.707107, y: 0.0, z: -0.707107 },
            Vertex { x: -0.83147, y: 0.0, z: -0.55557 },
            Vertex { x: -0.92388, y: 0.0, z: -0.382683 },
            Vertex { x: -0.980785, y: 0.0, z: -0.19509 },
            Vertex { x: -1.0, y: 0.0, z: 0.0 },
            Vertex { x: -0.980785, y: 0.0, z: 0.19509 },
            Vertex { x: -0.92388, y: 0.0, z: 0.382683 },
            Vertex { x: -0.83147, y: 0.0, z: 0.55557 },
            Vertex { x: -0.707107, y: 0.0, z: 0.707107 },
            Vertex { x: -0.55557, y: 0.0, z: 0.83147 },
            Vertex { x: -0.382683, y: 0.0, z: 0.92388 },
            Vertex { x: -0.19509, y: 0.0, z: 0.980785 },
            Vertex { x: 0.0, y: 0.0, z: 1.0 },
            Vertex { x: 0.195091, y: 0.0, z: 0.980785 },
            Vertex { x: 0.382684, y: 0.0, z: 0.923879 },
            Vertex { x: 0.555571, y: 0.0, z: 0.831469 },
            Vertex { x: 0.707107, y: 0.0, z: 0.707106 },
            Vertex { x: 0.83147, y: 0.0, z: 0.55557 },
            Vertex { x: 0.92388, y: 0.0, z: 0.382683 },
            Vertex { x: 0.980785, y: 0.0, z: 0.195089 },
            Vertex { x: 1.0, y: 0.0, z: -0.000001 },
            Vertex { x: 0.980785, y: 0.0, z: -0.195091 },
            Vertex { x: 0.923879, y: 0.0, z: -0.382684 },
            Vertex { x: 0.831469, y: 0.0, z: -0.555571 },
            Vertex { x: 0.707106, y: 0.0, z: -0.707108 },
            Vertex { x: 0.555569, y: 0.0, z: -0.83147 },
            Vertex { x: 0.382682, y: 0.0, z: -0.92388 },
            Vertex { x: 0.195089, y: 0.0, z: -0.980786 }
          ),
          geometry: vec!(
            Geometry {
              material_name: None,
              use_smooth_shading: false,
              shapes: vec!(
                Line(1, 0),
                Line(2, 1),
                Line(3, 2),
                Line(4, 3),
                Line(5, 4),
                Line(6, 5),
                Line(7, 6),
                Line(8, 7),
                Line(9, 8),
                Line(10, 9),
                Line(11, 10),
                Line(12, 11),
                Line(13, 12),
                Line(14, 13),
                Line(15, 14),
                Line(16, 15),
                Line(17, 16),
                Line(18, 17),
                Line(19, 18),
                Line(20, 19),
                Line(21, 20),
                Line(22, 21),
                Line(23, 22),
                Line(24, 23),
                Line(25, 24),
                Line(26, 25),
                Line(27, 26),
                Line(28, 27),
                Line(29, 28),
                Line(30, 29),
                Line(31, 30),
                Line(0, 31),
              )
            }
          )
        },
        Object {
          name: "Cube".into_string(),
          verticies: vec!(
            Vertex { x: 1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: 1.0, z: -0.999999 },
            Vertex { x: 0.999999, y: 1.0, z: 1.000001 },
            Vertex { x: -1.0, y: 1.0, z: 1.0 },
            Vertex { x: -1.0, y: 1.0, z: -1.0 }
          ),
          geometry: vec!(
            Geometry {
              material_name: Some("Material".into_string()),
              use_smooth_shading: false,
              shapes: vec!(
                Triangle(3, 0, 1),
                Triangle(3, 1, 2),
                Triangle(5, 4, 7),
                Triangle(5, 7, 6),
                Triangle(1, 0, 4),
                Triangle(1, 4, 5),
                Triangle(2, 1, 5),
                Triangle(2, 5, 6),
                Triangle(3, 2, 6),
                Triangle(3, 6, 7),
                Triangle(7, 4, 0),
                Triangle(7, 0, 3),
              )
            }
          )
        }
      )
    });

  let mut p = Parser::new(test_case);

  assert_eq!(p.parse_objset(), expected);
}

/// Parses a wavefront `.obj` file, returning either the successfully parsed
/// file, or an error. Support in this parser for the full file format is
/// best-effort and realistically I will only end up supporting the subset
/// of the file format which falls under the "shit I see exported from blender"
/// category.
pub fn parse(input: &str) -> Result<ObjSet, ParseError> {
  Parser::new(input).parse_objset()
}
