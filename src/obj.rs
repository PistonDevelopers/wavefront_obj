//! A parser for Wavefront's `.obj` file format for storing 3D meshes.
use std::iter;
use std::mem;
use std::result::{Result,Ok,Err};

use lex::{ParseError,Lexer};

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
  /// The set of vertices this object is composed of. These are referenced
  /// by index in `faces`.
  pub vertices: Vec<Vertex>,
  /// The set of texture vertices referenced by this object. The actual
  /// vertices are indexed by the second element in a `VTIndex`.
  pub tex_vertices: Vec<TVertex>,
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
  Point(VTIndex),
  /// A line specified by its endpoints.
  Line(VTIndex, VTIndex),
  /// A triangle specified by its three vertices.
  Triangle(VTIndex, VTIndex, VTIndex),
}

/// A single 3-dimensional point on the corner of an object.
#[allow(missing_docs)]
#[deriving(Clone, Copy, Show)]
pub struct Vertex {
  pub x: f64,
  pub y: f64,
  pub z: f64,
}

/// A single 2-dimensional point on a texture. "Texure Vertex".
#[allow(missing_docs)]
#[deriving(Clone, Copy, Show)]
pub struct TVertex {
  pub x: f64,
  pub y: f64,
}

fn fuzzy_cmp(a: f64, b: f64, delta: f64) -> Ordering {
  if (a - b).abs() <= delta {
    Equal
  } else if a < b {
    Less
  } else {
    Greater
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

impl PartialEq for TVertex {
  fn eq(&self, other: &TVertex) -> bool {
    self.partial_cmp(other).unwrap() == Equal
  }
}

impl PartialOrd for TVertex {
  fn partial_cmp(&self, other: &TVertex) -> Option<Ordering> {
    Some(fuzzy_cmp(self.x, other.x, 0.00001)
      .cmp(&fuzzy_cmp(self.y, other.y, 0.00001)))
  }
}

/// An index into the `vertices` array of an object, representing a vertex in
/// the mesh. After parsing, this is guaranteed to be a valid index into the
/// array, so unchecked indexing may be used.
pub type VertexIndex = uint;

/// An index into the `texture vertex` array of an object.
///
/// Unchecked indexing may be used, because the values are guaranteed to be in
/// range by the parser.
pub type TextureIndex = uint;

/// An index into the vertex array, with an optional index into the texture
/// array. This is used to define the corners of shapes which may or may not
/// be textured.
pub type VTIndex = (VertexIndex, Option<TextureIndex>);

/// Slices the underlying string in an option.
fn sliced<'a>(s: &'a Option<String>) -> Option<&'a str> {
  match *s {
    None => None,
    Some(ref s) => Some(s.as_slice()),
  }
}

/// Blender exports shapes as a list of the vertices representing their corners.
/// This function turns that into a set of OpenGL-usable shapes - i.e. points,
/// lines, or triangles.
fn to_triangles(xs: &[VTIndex]) -> Vec<Shape> {
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

  assert_eq!(to_triangles(&[(3,None)]), vec!(Point((3,None))));

  assert_eq!(
    to_triangles(&[
      (1,None)
      ,(2,None)
    ]),
    vec!(
      Line(
        (1,None),
        (2,None)
      )
    ));

  assert_eq!(
    to_triangles(&[
      (1,None),
      (2,None),
      (3,None)
    ]),
    vec!(
      Triangle(
        (3,None),
        (1,None),
        (2,None)
      )
    ));

  assert_eq!(
    to_triangles(&[
      (1,None),
      (2,None),
      (3,None),
      (4,None)
    ]),
    vec!(
      Triangle(
        (4,None),
        (1,None),
        (2,None)),
      Triangle(
        (4,None),
        (2,None),
        (3,None)
      )
    ));

  assert_eq!(
    to_triangles(&[
      (1,None),
      (2,None),
      (3,None),
      (4,None),
      (5,None)
    ]), vec!(
      Triangle(
        (5,None),
        (1,None),
        (2,None)),
      Triangle(
        (5,None),
        (2,None),
        (3,None)),
      Triangle(
        (5,None),
        (3,None),
        (4,None)
      )
    ));
}

struct Parser<'a> {
  line_number: uint,
  lexer: iter::Peekable<String, Lexer<'a>>,
}

impl<'a> Parser<'a> {
  fn new<'a>(input: &'a str) -> Parser<'a> {
    Parser {
      line_number: 1,
      lexer: Lexer::new(input).peekable(),
    }
  }

  fn error<A>(&self, msg: String) -> Result<A, ParseError> {
    Err(ParseError {
      line_number: self.line_number,
      message:     msg,
    })
  }

  fn next(&mut self) -> Option<String> {
    // TODO(cgaebel): This has a lot of useless allocations. Techincally we can
    // just be using slices into the underlying buffer instead of allocating a
    // new string for every single token. Unfortunately, I'm not sure how to
    // structure this to appease the borrow checker.
    let ret = self.lexer.next();

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

  /// Parse just a constant string.
  fn parse_tag(&mut self, tag: &str) -> Result<(), ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error(format!("Expected `{}` but got end of input.", tag)),
      Some(s) => {
        if s != tag {
          return self.error(format!("Expected `{}` but got {}.", tag, s));
        }
      }
    }

    return Ok(())
  }

  /// Skips over some newlines, failing if it didn't manage to skip any.
  fn one_or_more_newlines(&mut self) -> Result<(), ParseError> {
    try!(self.parse_tag("\n"));
    self.zero_or_more_newlines();
    Ok(())
  }

  fn parse_str(&mut self) -> Result<String, ParseError> {
    match self.next() {
      None =>
        self.error(format!("Expected string but got end of input.")),
      Some(got) => {
        if got.as_slice() == "\n" {
          self.error(format!("Expected string but got `end of line`."))
        } else {
          Ok(got)
        }
      }
    }
  }

  fn parse_material_library<'a>(&mut self) -> Result<String, ParseError> {
    try!(self.parse_tag("mtllib"));
    self.parse_str()
  }

  fn parse_object_name(&mut self) -> Result<String, ParseError> {
    try!(self.parse_tag("o"));
    self.parse_str()
  }

  // TODO(cgaebel): Should this be returning `num::rational::BigRational` instead?
  // I can't think of a good reason to do this except to make testing easier.
  fn parse_double(&mut self) -> Result<f64, ParseError> {
    let s = try!(self.parse_str());

    match from_str::<f64>(s.as_slice()) {
      None =>
        self.error(format!("Expected f64 but got {}.", s)),
      Some(ret) =>
        Ok(ret)
    }
  }

  fn parse_vertex(&mut self) -> Result<Vertex, ParseError> {
    try!(self.parse_tag("v"));

    let x = try!(self.parse_double());
    let y = try!(self.parse_double());
    let z = try!(self.parse_double());

    Ok(Vertex { x: x, y: y, z: z })
  }

  /// BUG: Also munches trailing whitespace.
  fn parse_vertices(&mut self) -> Result<Vec<Vertex>, ParseError> {
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

  fn parse_tex_vertex(&mut self) -> Result<TVertex, ParseError> {
    try!(self.parse_tag("vt"));

    let x = try!(self.parse_double());
    let y = try!(self.parse_double());

    Ok(TVertex { x: x, y: y })
  }

  /// BUG: Also munches trailing whitespace.
  fn parse_tex_vertices(&mut self) -> Result<Vec<TVertex>, ParseError> {
    let mut result = Vec::new();

    loop {
      match sliced(&self.peek()) {
        Some("vt") => {
          let v = try!(self.parse_tex_vertex());
          result.push(v);
        },
        _ => break,
      }

      try!(self.one_or_more_newlines());
    }

    Ok(result)
  }

  fn parse_usemtl(&mut self) -> Result<String, ParseError> {
    try!(self.parse_tag("usemtl"));
    self.parse_str()
  }

  fn parse_smooth_shading(&mut self) -> Result<bool, ParseError> {
    try!(self.parse_tag("s"));

    match try!(self.parse_str()).as_slice() {
      "on"  => Ok(true),
      "off" => Ok(false),
      s     => self.error(format!("Expected `on` or `off` but got {}.", s)),
    }
  }

  fn parse_int_from(&mut self, s: &str) -> Result<int, ParseError> {
    match from_str::<int>(s) {
      None =>
        return self.error(format!("Expected int but got {}.", s)),
      Some(ret) =>
        Ok(ret)
    }
  }

  fn parse_vtindex(&mut self, valid_vtx: (uint, uint), valid_tx: (uint, uint)) -> Result<VTIndex, ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error("Expected vertex index but got end of input.".into_string()),
      Some(s) => {
        let splits: Vec<&str> = s.split('/').collect();
        assert!(splits.len() != 0);

        match splits.len() {
          1 => {
            let v_idx = try!(self.parse_int_from(splits[0]));
            let v_idx = try!(self.check_valid_index(valid_vtx, v_idx));
            Ok((v_idx, None))
          },
          2 => {
            let v_idx = try!(self.parse_int_from(splits[0]));
            let v_idx = try!(self.check_valid_index(valid_vtx, v_idx));
            let t_idx = try!(self.parse_int_from(splits[1]));
            let t_idx = try!(self.check_valid_index(valid_tx, t_idx));
            Ok((v_idx, Some(t_idx)))
          },
          n =>
            self.error(format!("Expected at most 2 vertex indexes but got {}.", n)),
        }
      }
    }
  }

  /// `valid_values` is a range of valid bounds for the actual value.
  fn check_valid_index(&self, valid_values: (uint, uint), actual_value: int) -> Result<uint, ParseError> {
    let (min, max) = valid_values;

    let mut x = actual_value;

    // Handle negative vertex indexes.
    if x < 0 {
      x = max as int - x;
    }

    if x >= min as int && x < max as int {
      assert!(x > 0);
      Ok((x - min as int) as uint)
    } else {
      self.error(format!("Expected index in the range [{}, {}), but got {}.", min, max, actual_value))
    }
  }

  fn parse_face(&mut self, valid_vtx: (uint, uint), valid_tx: (uint, uint)) -> Result<Vec<Shape>, ParseError> {
    match sliced(&self.next()) {
      Some("f") => {},
      Some("l") => {},
      None      => return self.error("Expected `f` or `l` but got end of input.".into_string()),
      Some(s)   => return self.error(format!("Expected `f` or `l` but got {}.", s)),
    }

    let mut corner_list = Vec::new();

    corner_list.push(try!(self.parse_vtindex(valid_vtx, valid_tx)));

    loop {
      match sliced(&self.peek()) {
        None       => break,
        Some("\n") => break,
        Some( _  ) => corner_list.push(try!(self.parse_vtindex(valid_vtx, valid_tx))),
      }
    }

    Ok(to_triangles(corner_list.as_slice()))
  }

  fn parse_geometries(&mut self, valid_vtx: (uint, uint), valid_tx: (uint, uint)) -> Result<Vec<Geometry>, ParseError> {
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
          shapes.push_all(try!(self.parse_face(valid_vtx, valid_tx)).as_slice());
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

    Ok(result.into_iter().filter(|ref x| !x.shapes.is_empty()).collect())
  }

  fn parse_object(&mut self,
      min_vertex_index: &mut uint,
      max_vertex_index: &mut uint,
      min_tex_index:    &mut uint,
      max_tex_index:    &mut uint) -> Result<Object, ParseError> {
    let name = try!(self.parse_object_name());
    try!(self.one_or_more_newlines());

    let vertices     = try!(self.parse_vertices());
    let tex_vertices = try!(self.parse_tex_vertices());

    *max_vertex_index += vertices.len();
    *max_tex_index    += tex_vertices.len();

    let geometry =
      try!(self.parse_geometries(
        (*min_vertex_index, *max_vertex_index),
        (*min_tex_index, *max_tex_index)));

    *min_vertex_index += vertices.len();
    *min_tex_index    += tex_vertices.len();

    Ok(Object {
      name:          name,
      vertices:     vertices,
      tex_vertices: tex_vertices,
      geometry:      geometry,
    })
  }

  fn parse_objects(&mut self) -> Result<Vec<Object>, ParseError> {
    let mut result = Vec::new();

    let mut min_vertex_index = 1;
    let mut max_vertex_index = 1;
    let mut min_tex_index    = 1;
    let mut max_tex_index    = 1;

    loop {
      match sliced(&self.peek()) {
        Some("o") => result.push(try!(self.parse_object(
                      &mut min_vertex_index,
                      &mut max_vertex_index,
                      &mut min_tex_index,
                      &mut max_tex_index))),
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
          vertices: vec!(
            Vertex { x: -1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: 1.0, z: 1.0 },
            Vertex { x: -1.0, y: 1.0, z: -1.0 },
            Vertex { x: 1.0, y: 1.0, z: -1.0 },
            Vertex { x: 1.0, y: 1.0, z: 1.0 }
          ),
          tex_vertices: vec!(),
          geometry: vec!(
            Geometry {
              material_name: Some("None".into_string()),
              use_smooth_shading: false,
              shapes: vec!(
                Triangle((0, None), (4, None), (5, None)),
                Triangle((0, None), (5, None), (1, None)),
                Triangle((1, None), (5, None), (6, None)),
                Triangle((1, None), (6, None), (2, None)),
                Triangle((2, None), (6, None), (7, None)),
                Triangle((2, None), (7, None), (3, None)),
                Triangle((3, None), (7, None), (4, None)),
                Triangle((3, None), (4, None), (0, None)),
                Triangle((3, None), (0, None), (1, None)),
                Triangle((3, None), (1, None), (2, None)),
                Triangle((4, None), (7, None), (6, None)),
                Triangle((4, None), (6, None), (5, None)),
              )
            }
          )
        },
        Object {
          name: "Circle".into_string(),
          vertices: vec!(
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
          tex_vertices: vec!(),
          geometry: vec!(
            Geometry {
              material_name: None,
              use_smooth_shading: false,
              shapes: vec!(
                Line((1, None), (0, None)),
                Line((2, None), (1, None)),
                Line((3, None), (2, None)),
                Line((4, None), (3, None)),
                Line((5, None), (4, None)),
                Line((6, None), (5, None)),
                Line((7, None), (6, None)),
                Line((8, None), (7, None)),
                Line((9, None), (8, None)),
                Line((10, None), (9, None)),
                Line((11, None), (10, None)),
                Line((12, None), (11, None)),
                Line((13, None), (12, None)),
                Line((14, None), (13, None)),
                Line((15, None), (14, None)),
                Line((16, None), (15, None)),
                Line((17, None), (16, None)),
                Line((18, None), (17, None)),
                Line((19, None), (18, None)),
                Line((20, None), (19, None)),
                Line((21, None), (20, None)),
                Line((22, None), (21, None)),
                Line((23, None), (22, None)),
                Line((24, None), (23, None)),
                Line((25, None), (24, None)),
                Line((26, None), (25, None)),
                Line((27, None), (26, None)),
                Line((28, None), (27, None)),
                Line((29, None), (28, None)),
                Line((30, None), (29, None)),
                Line((31, None), (30, None)),
                Line((0, None), (31, None)),
              )
            }
          )
        },
        Object {
          name: "Cube".into_string(),
          vertices: vec!(
            Vertex { x: 1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: -1.0, z: 1.0 },
            Vertex { x: -1.0, y: -1.0, z: -1.0 },
            Vertex { x: 1.0, y: 1.0, z: -0.999999 },
            Vertex { x: 0.999999, y: 1.0, z: 1.000001 },
            Vertex { x: -1.0, y: 1.0, z: 1.0 },
            Vertex { x: -1.0, y: 1.0, z: -1.0 }
          ),
          tex_vertices: vec!(),
          geometry: vec!(
            Geometry {
              material_name: Some("Material".into_string()),
              use_smooth_shading: false,
              shapes: vec!(
                Triangle((3, None), (0, None), (1, None)),
                Triangle((3, None), (1, None), (2, None)),
                Triangle((5, None), (4, None), (7, None)),
                Triangle((5, None), (7, None), (6, None)),
                Triangle((1, None), (0, None), (4, None)),
                Triangle((1, None), (4, None), (5, None)),
                Triangle((2, None), (1, None), (5, None)),
                Triangle((2, None), (5, None), (6, None)),
                Triangle((3, None), (2, None), (6, None)),
                Triangle((3, None), (6, None), (7, None)),
                Triangle((7, None), (4, None), (0, None)),
                Triangle((7, None), (0, None), (3, None)),
              )
            }
          )
        }
      )
    });

  assert_eq!(parse(test_case.into_string()), expected);
}

#[test]
fn test_cube() {
  let test_case =
r#"
# Blender v2.71 (sub 0) OBJ File: 'cube.blend'
# www.blender.org
mtllib cube.mtl
o Cube
v 1.000000 -1.000000 -1.000000
v 1.000000 -1.000000 1.000000
v -1.000000 -1.000000 1.000000
v -1.000000 -1.000000 -1.000000
v 1.000000 1.000000 -0.999999
v 0.999999 1.000000 1.000001
v -1.000000 1.000000 1.000000
v -1.000000 1.000000 -1.000000
vt 1.004952 0.498633
vt 0.754996 0.498236
vt 0.755393 0.248279
vt 1.005349 0.248677
vt 0.255083 0.497442
vt 0.255480 0.247485
vt 0.505437 0.247882
vt 0.505039 0.497839
vt 0.754598 0.748193
vt 0.504642 0.747795
vt 0.505834 -0.002074
vt 0.755790 -0.001677
vt 0.005127 0.497044
vt 0.005524 0.247088
usemtl Material
s off
f 1/1 2/2 3/3 4/4
f 5/5 8/6 7/7 6/8
f 1/9 5/10 6/8 2/2
f 2/2 6/8 7/7 3/3
f 3/3 7/7 8/11 4/12
f 5/5 1/13 4/14 8/6
"#;

  let expected =
    Ok(ObjSet {
      material_library: "cube.mtl".into_string(),
      objects: vec![
        Object {
          name: "Cube".into_string(),
          vertices: vec![
            Vertex { x:  1.0, y: -1.0, z: -1.0 },
            Vertex { x:  1.0, y: -1.0, z:  1.0 },
            Vertex { x: -1.0, y: -1.0, z:  1.0 },
            Vertex { x: -1.0, y: -1.0, z: -1.0 },
            Vertex { x:  1.0, y:  1.0, z: -1.0 },
            Vertex { x:  1.0, y:  1.0, z:  1.0 },
            Vertex { x: -1.0, y:  1.0, z:  1.0 },
            Vertex { x: -1.0, y:  1.0, z: -1.0 }
          ],
          tex_vertices: vec![
            TVertex { x: 1.004952, y: 0.498633 },
            TVertex { x: 0.754996, y: 0.498236 },
            TVertex { x: 0.755393, y: 0.248279 },
            TVertex { x: 1.005349, y: 0.248677 },
            TVertex { x: 0.255083, y: 0.497442 },
            TVertex { x: 0.25548, y: 0.247485 },
            TVertex { x: 0.505437, y: 0.247882 },
            TVertex { x: 0.505039, y: 0.497839 },
            TVertex { x: 0.754598, y: 0.748193 },
            TVertex { x: 0.504642, y: 0.747795 },
            TVertex { x: 0.505834, y: -0.002074 },
            TVertex { x: 0.75579, y: -0.001677 },
            TVertex { x: 0.005127, y: 0.497044 },
            TVertex { x: 0.005524, y: 0.247088 }
          ],
          geometry: vec![
            Geometry {
              material_name: Some("Material".into_string()),
              use_smooth_shading: false,
              shapes: vec![
                Triangle((3, Some(3)), (0, Some(0)), (1, Some(1))),
                Triangle((3, Some(3)), (1, Some(1)), (2, Some(2))),
                Triangle((5, Some(7)), (4, Some(4)), (7, Some(5))),
                Triangle((5, Some(7)), (7, Some(5)), (6, Some(6))),
                Triangle((1, Some(1)), (0, Some(8)), (4, Some(9))),
                Triangle((1, Some(1)), (4, Some(9)), (5, Some(7))),
                Triangle((2, Some(2)), (1, Some(1)), (5, Some(7))),
                Triangle((2, Some(2)), (5, Some(7)), (6, Some(6))),
                Triangle((3, Some(11)), (2, Some(2)), (6, Some(6))),
                Triangle((3, Some(11)), (6, Some(6)), (7, Some(10))),
                Triangle((7, Some(5)), (4, Some(4)), (0, Some(12))),
                Triangle((7, Some(5)), (0, Some(12)), (3, Some(13)))
              ]
            }
          ]
        }
      ]
    });

  assert_eq!(parse(test_case.into_string()), expected);
}

/// Parses a wavefront `.obj` file, returning either the successfully parsed
/// file, or an error. Support in this parser for the full file format is
/// best-effort and realistically I will only end up supporting the subset
/// of the file format which falls under the "things I see exported from blender"
/// category.
pub fn parse(mut input: String) -> Result<ObjSet, ParseError> {
  // Unfortunately, the parser requires a trailing newline. This is the easiest
  // way I could find to allow non-trailing newlines.
  input.push_str("\n");
  Parser::new(input.as_slice()).parse_objset()
}
