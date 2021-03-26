//! A parser for Wavefront's `.obj` file format for storing 3D meshes.
use lex::PeekableLexer;
use std::borrow::ToOwned;
use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::mem;

use lex::{Lexer, ParseError};
use util::OrderingExt;

/// A set of objects, as listed in an `.obj` file.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjSet {
  /// Which material library to use.
  pub material_library: Option<String>,
  /// The set of objects.
  pub objects: Vec<Object>,
}

/// A mesh object.
#[derive(Clone, Debug, PartialEq)]
pub struct Object {
  /// A human-readable name for this object.
  pub name: String,
  /// The set of vertices this object is composed of. These are referenced
  /// by index in `shapes` contained within each element of `geometry`.
  pub vertices: Vec<Vertex>,
  /// The set of texture vertices referenced by this object. The actual
  /// vertices are indexed by the second element in a `VTNIndex`.
  pub tex_vertices: Vec<TVertex>,
  /// The set of normals referenced by this object. This are are referenced
  /// by the third element in a `VTNIndex`.
  pub normals: Vec<Normal>,
  /// A set of shapes (with materials applied to them) of which this object is
  /// composed.
  pub geometry: Vec<Geometry>,
}

/// A set of shapes, all using the given material.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
  /// A reference to the material to apply to this geometry.
  pub material_name: Option<String>,
  /// The shapes of which this geometry is composed.
  pub shapes: Vec<Shape>,
}

/// A shape gathers a primitive and groups.
///
/// Each shape is associated with 0 or many groups. Those are text identifiers
/// used to gather geometry elements into different groups.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
  /// The primitive of the shape.
  pub primitive: Primitive,
  /// Associated groups. No associated group means the shape uses the default
  /// group.
  pub groups: Vec<GroupName>,
  /// Associated smoothing groups. No associated smoothing group means the shape should be rendered
  /// flat.
  pub smoothing_groups: Vec<u32>,
}

/// Name of a group.
pub type GroupName = String;

/// The various primitives supported by this library.
///
/// Convex polygons more complicated than a triangle are automatically
/// converted into triangles.
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub enum Primitive {
  /// A point specified by its position.
  Point(VTNIndex),
  /// A line specified by its endpoints.
  Line(VTNIndex, VTNIndex),
  /// A triangle specified by its three vertices.
  Triangle(VTNIndex, VTNIndex, VTNIndex),
}

/// A single 3-dimensional point on the corner of an object.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
  pub x: f64,
  pub y: f64,
  pub z: f64,
}

/// A single 3-dimensional normal
pub type Normal = Vertex;

/// A single 3-dimensional point on a texture. "Texure Vertex".
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug)]
pub struct TVertex {
  pub u: f64,
  pub v: f64,
  pub w: f64,
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
    Some(
      fuzzy_cmp(self.x, other.x, 0.00001)
        .lexico(|| fuzzy_cmp(self.y, other.y, 0.00001))
        .lexico(|| fuzzy_cmp(self.z, other.z, 0.00001)),
    )
  }
}

impl PartialEq for TVertex {
  fn eq(&self, other: &TVertex) -> bool {
    self.partial_cmp(other).unwrap() == Equal
  }
}

impl PartialOrd for TVertex {
  fn partial_cmp(&self, other: &TVertex) -> Option<Ordering> {
    Some(
      fuzzy_cmp(self.u, other.u, 0.00001)
        .lexico(|| fuzzy_cmp(self.v, other.v, 0.00001))
        .lexico(|| fuzzy_cmp(self.w, other.w, 0.00001)),
    )
  }
}

/// An index into the `vertices` array of an object, representing a vertex in
/// the mesh. After parsing, this is guaranteed to be a valid index into the
/// array, so unchecked indexing may be used.
pub type VertexIndex = usize;

/// An index into the `texture vertex` array of an object.
///
/// Unchecked indexing may be used, because the values are guaranteed to be in
/// range by the parser.
pub type TextureIndex = usize;

/// An index into the `normals` array of an object.
///
/// Unchecked indexing may be used, because the values are guaranteed to be in
/// range by the parser.
pub type NormalIndex = usize;

/// An index into the vertex array, with an optional index into the texture
/// array. This is used to define the corners of shapes which may or may not
/// be textured.
pub type VTNIndex = (VertexIndex, Option<TextureIndex>, Option<NormalIndex>);

/// Blender exports primitives as a list of the vertices representing their corners.
/// This function turns that into a set of OpenGL-usable shapes - i.e. points,
/// lines, or triangles.
fn to_triangles(xs: &[VTNIndex]) -> Vec<Primitive> {
  match xs.len() {
    0 => return vec![],
    1 => return vec![Primitive::Point(xs[0])],
    2 => return vec![Primitive::Line(xs[0], xs[1])],
    _ => {}
  }

  let last_elem = *xs.last().unwrap();

  xs[..xs.len() - 1]
    .iter()
    .zip(xs[1..xs.len() - 1].iter())
    .map(|(&x, &y)| Primitive::Triangle(last_elem, x, y))
    .collect()
}

#[test]
fn test_to_triangles() {
  use self::Primitive::{Line, Point, Triangle};

  assert_eq!(to_triangles(&[]), vec!());

  assert_eq!(
    to_triangles(&[(3, None, None)]),
    vec!(Point((3, None, None)))
  );

  assert_eq!(
    to_triangles(&[(1, None, None), (2, None, None)]),
    vec!(Line((1, None, None), (2, None, None)))
  );

  assert_eq!(
    to_triangles(&[(1, None, None), (2, None, None), (3, None, None)]),
    vec!(Triangle((3, None, None), (1, None, None), (2, None, None)))
  );

  assert_eq!(
    to_triangles(&[
      (1, None, None),
      (2, None, None),
      (3, None, None),
      (4, None, None)
    ]),
    vec!(
      Triangle((4, None, None), (1, None, None), (2, None, None)),
      Triangle((4, None, None), (2, None, None), (3, None, None))
    )
  );

  assert_eq!(
    to_triangles(&[
      (1, None, None),
      (2, None, None),
      (3, None, None),
      (4, None, None),
      (5, None, None)
    ]),
    vec!(
      Triangle((5, None, None), (1, None, None), (2, None, None)),
      Triangle((5, None, None), (2, None, None), (3, None, None)),
      Triangle((5, None, None), (3, None, None), (4, None, None))
    )
  );
}

#[derive(Clone)]
struct Parser<'a> {
  line_number: usize,
  lexer: PeekableLexer<'a>,
}

impl<'a> Parser<'a> {
  fn new(input: &'a str) -> Parser<'a> {
    Parser {
      line_number: 1,
      lexer: PeekableLexer::new(Lexer::new(input)),
    }
  }

  fn error_raw(&self, msg: String) -> ParseError {
    ParseError {
      line_number: self.line_number,
      message: msg,
    }
  }

  fn error<A, E>(&self, msg: E) -> Result<A, ParseError>
  where
    E: Into<String>,
  {
    Err(self.error_raw(msg.into()))
  }

  fn next(&mut self) -> Option<&'a str> {
    let ret = self.lexer.next_str();
    if let Some("\n") = ret {
      self.line_number += 1;
    }
    ret
  }

  fn advance(&mut self) {
    self.next();
  }

  fn peek(&mut self) -> Option<&'a str> {
    self.lexer.peek_str()
  }

  /// Take a parser function and try to parse with it. If the parser fails, `None` is returned and
  /// no input is consumed. If the parser succeeds, the input is consumed and the parser result
  /// is returned.
  ///
  /// Be careful while using this function, especially in recursive parsing as it might end up with
  /// non-linear parsing.
  fn try<P, T>(&mut self, parse: P) -> Option<T>
  where
    P: FnOnce(&mut Self) -> Result<T, ParseError>,
  {
    let mut tried = self.clone();

    match parse(&mut tried) {
      Ok(r) => {
        *self = tried;
        Some(r)
      }
      Err(_) => None,
    }
  }

  /// Return the number of bytes read since the last lexer state provided as argument.
  ///
  /// Return [`None`] if the checkpoint parser is more advanced in the input than the current one —
  /// i.e. you have likely swapped the parsers, try calling the other way around!
  fn bytes_consumed(&self, checkpoint: &Self) -> Option<usize> {
    self.lexer.bytes_consumed(&checkpoint.lexer)
  }

  /// Possibly skips over some newlines.
  fn zero_or_more_newlines(&mut self) {
    while let Some("\n") = self.peek() {
      self.advance()
    }
  }

  /// Parse just a constant string.
  fn parse_tag(&mut self, tag: &str) -> Result<(), ParseError> {
    match self.next() {
      None => self.error(format!("Expected `{}` but got end of input.", tag)),
      Some(s) if s != tag => self.error(format!("Expected `{}` but got {}.", tag, s)),
      _ => Ok(()),
    }
  }

  fn parse_tag_or_eof(&mut self, tag: &str) -> Result<(), ParseError> {
    match self.next() {
      Some(s) if s != tag => self.error(format!("Expected `{}` or EOF but got {}.", tag, s)),
      _ => Ok(()),
    }
  }

  /// Skips over some newlines, failing if it didn't manage to skip any.
  fn one_or_more_newlines(&mut self) -> Result<(), ParseError> {
    self.parse_tag_or_eof("\n")?;
    self.zero_or_more_newlines();
    Ok(())
  }

  fn parse_str(&mut self) -> Result<&'a str, ParseError> {
    match self.next() {
      None => self.error(format!("Expected string but got end of input.")),
      Some("\n") => self.error(format!("Expected string but got `end of line`.")),
      Some(got) => Ok(got),
    }
  }

  fn parse_material_library(&mut self) -> Result<Option<&'a str>, ParseError> {
    match self.peek() {
      Some("mtllib") => {}
      _ => return Ok(None),
    }
    self.advance();
    self.parse_str().map(Some)
  }

  fn parse_object_name(&mut self) -> Result<&'a str, ParseError> {
    match self.peek() {
      Some("o") => {
        self.parse_tag("o")?;
        let name = self.parse_str();
        self.one_or_more_newlines()?;
        name
      }
      _ => Ok(""),
    }
  }

  // TODO(cgaebel): Should this be returning `num::rational::BigRational` instead?
  // I can't think of a good reason to do this except to make testing easier.
  fn parse_double(&mut self) -> Result<f64, ParseError> {
    let s = self.parse_str()?;
    lexical::parse(s).map_err(|_| self.error_raw(format!("Expected f64 but got {}.", s)))
  }

  fn parse_vertex(&mut self) -> Result<Vertex, ParseError> {
    self.parse_tag("v")?;

    let x = self.parse_double()?;
    let y = self.parse_double()?;
    let z = self.parse_double()?;

    Ok(Vertex { x, y, z })
  }

  fn parse_tex_vertex(&mut self) -> Result<TVertex, ParseError> {
    self.parse_tag("vt")?;
    let u = self.parse_double()?;

    match self.try(Self::parse_double) {
      Some(v) => {
        let w = self.try(Self::parse_double).unwrap_or(0.);
        Ok(TVertex { u, v, w })
      }
      None => Ok(TVertex { u, v: 0., w: 0. }),
    }
  }

  fn parse_normal(&mut self) -> Result<Normal, ParseError> {
    self.parse_tag("vn")?;

    let x = self.parse_double()?;
    let y = self.parse_double()?;
    let z = self.parse_double()?;

    Ok(Normal { x, y, z })
  }

  fn parse_usemtl(&mut self) -> Result<&'a str, ParseError> {
    self.parse_tag("usemtl")?;
    self.parse_str()
  }

  #[inline]
  fn parse_isize_from(&self, s: &str) -> Result<isize, ParseError> {
    lexical::parse(&s).map_err(|_| self.error_raw(format!("Expected isize but got {}.", s)))
  }

  fn parse_u32(&mut self) -> Result<u32, ParseError> {
    let s = self.parse_str()?;
    lexical::parse(&s).map_err(|_| self.error_raw(format!("Expected u32 but got {}.", s)))
  }

  fn parse_vtindex(
    &mut self,
    valid_vtx: (usize, usize),
    valid_tx: (usize, usize),
    valid_nx: (usize, usize),
  ) -> Result<VTNIndex, ParseError> {
    match self.next() {
      None => return self.error("Expected vertex index but got end of input.".to_owned()),
      Some(s) => {
        let process_split =
          |split: &str, valid_range: (usize, usize)| -> Result<Option<usize>, ParseError> {
            if split.len() > 0 {
              Ok(Some(self.check_valid_index(
                valid_range,
                self.parse_isize_from(split)?,
              )?))
            } else {
              Ok(None)
            }
          };

        let mut splits_iter = s.split('/');
        let split1 = splits_iter
          .next()
          .and_then(|s| process_split(&s, valid_vtx).transpose())
          .transpose()?;
        let split2 = splits_iter
          .next()
          .and_then(|s| process_split(&s, valid_tx).transpose())
          .transpose()?;
        let split3 = splits_iter
          .next()
          .and_then(|s| process_split(&s, valid_nx).transpose())
          .transpose()?;

        if split1.is_none() || splits_iter.next().is_some() {
          self.error(format!("Expected at least 1 and at most 3 vertex indexes."))
        } else {
          Ok((split1.unwrap(), split2, split3))
        }
      }
    }
  }

  /// `valid_values` is a range of valid bounds for the actual value.
  #[inline(always)]
  fn check_valid_index(
    &self,
    valid_values: (usize, usize),
    actual_value: isize,
  ) -> Result<usize, ParseError> {
    let (min, max) = valid_values;

    let mut x = actual_value;

    // Handle negative vertex indexes.
    if x < 0 {
      x = max as isize - x;
    }

    if x >= min as isize && x < max as isize {
      debug_assert!(x > 0);
      Ok((x - min as isize) as usize)
    } else {
      self.error(format!(
        "Expected index in the range [{}, {}), but got {}.",
        min, max, actual_value
      ))
    }
  }

  fn parse_face(
    &mut self,
    valid_vtx: (usize, usize),
    valid_tx: (usize, usize),
    valid_nx: (usize, usize),
    current_groups: &Vec<GroupName>,
    current_smoothing_groups: &Vec<u32>,
  ) -> Result<Vec<Shape>, ParseError> {
    match self.next() {
      Some("f") => {}
      Some("l") => {}
      None => return self.error("Expected `f` or `l` but got end of input.".to_owned()),
      Some(s) => return self.error(format!("Expected `f` or `l` but got {}.", s)),
    }

    let mut corner_list = Vec::new();

    corner_list.push(self.parse_vtindex(valid_vtx, valid_tx, valid_nx)?);

    loop {
      match self.peek() {
        None => break,
        Some("\n") => break,
        Some(_) => corner_list.push(self.parse_vtindex(valid_vtx, valid_tx, valid_nx)?),
      }
    }

    Ok(
      to_triangles(&corner_list)
        .into_iter()
        .map(|prim| Shape {
          primitive: prim,
          groups: current_groups.clone(),
          smoothing_groups: current_smoothing_groups.clone(),
        })
        .collect(),
    )
  }

  fn parse_geometries(
    &mut self,
    valid_vtx: (usize, usize),
    valid_tx: (usize, usize),
    valid_nx: (usize, usize),
  ) -> Result<Vec<Geometry>, ParseError> {
    let mut result = Vec::new();
    let mut shapes = Vec::new();

    let mut current_material = None;
    let mut current_groups = Vec::new();
    let mut current_smoothing_groups = Vec::new();

    loop {
      match self.peek() {
        Some("usemtl") => {
          let old_material = mem::replace(&mut current_material, Some(self.parse_usemtl()?));

          result.push(Geometry {
            material_name: old_material.map(|s| s.to_owned()),
            shapes: mem::replace(&mut shapes, Vec::new()),
          });
        }
        Some("s") => {
          self.advance();
          current_smoothing_groups = self.parse_smoothing_groups()?;
        }
        Some("f") | Some("l") => {
          shapes.extend(
            self
              .parse_face(
                valid_vtx,
                valid_tx,
                valid_nx,
                &current_groups,
                &current_smoothing_groups,
              )?
              .into_iter(),
          );
        }
        Some("g") => {
          self.advance();
          let names = self.parse_groups()?;
          current_groups = names;
        }
        _ => break,
      }

      self.one_or_more_newlines()?;
    }

    result.push(Geometry {
      material_name: current_material.map(|s| s.to_owned()),
      shapes,
    });

    Ok(
      result
        .into_iter()
        .filter(|ref x| !x.shapes.is_empty())
        .collect(),
    )
  }

  fn parse_object(
    &mut self,
    min_vertex_index: &mut usize,
    max_vertex_index: &mut usize,
    min_tex_index: &mut usize,
    max_tex_index: &mut usize,
    min_normal_index: &mut usize,
    max_normal_index: &mut usize,
  ) -> Result<Object, ParseError> {
    let name = self.parse_object_name()?;

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut tex_vertices = Vec::new();

    // read vertices, normals and texture coordinates
    loop {
      if let Some(v) = self.try(Self::parse_vertex) {
        vertices.push(v);
      } else if let Some(vn) = self.try(Self::parse_normal) {
        normals.push(vn);
      } else if let Some(vt) = self.try(Self::parse_tex_vertex) {
        tex_vertices.push(vt);
      } else {
        break;
      }

      self.one_or_more_newlines()?;
    }

    *max_vertex_index += vertices.len();
    *max_tex_index += tex_vertices.len();
    *max_normal_index += normals.len();

    let geometry = self.parse_geometries(
      (*min_vertex_index, *max_vertex_index),
      (*min_tex_index, *max_tex_index),
      (*min_normal_index, *max_normal_index),
    )?;

    *min_vertex_index += vertices.len();
    *min_tex_index += tex_vertices.len();
    *min_normal_index += normals.len();

    Ok(Object {
      name: name.to_owned(),
      vertices,
      tex_vertices,
      normals,
      geometry,
    })
  }

  fn parse_objects(&mut self) -> Result<Vec<Object>, ParseError> {
    let mut result = Vec::new();

    let mut min_vertex_index = 1;
    let mut max_vertex_index = 1;
    let mut min_tex_index = 1;
    let mut max_tex_index = 1;
    let mut min_normal_index = 1;
    let mut max_normal_index = 1;

    loop {
      match self.peek() {
        Some(_) => {
          // create a checkpoint parser so that we can check if we consumed bytes
          let checkpoint = self.clone();

          result.push(self.parse_object(
            &mut min_vertex_index,
            &mut max_vertex_index,
            &mut min_tex_index,
            &mut max_tex_index,
            &mut min_normal_index,
            &mut max_normal_index,
          )?);

          if self.bytes_consumed(&checkpoint).unwrap_or(0) == 0 {
            return self.error("cannot parse corrupted data");
          }
        }
        None => break,
      }

      self.zero_or_more_newlines();
    }

    Ok(result)
  }

  fn parse_objset(&mut self) -> Result<ObjSet, ParseError> {
    self.zero_or_more_newlines();

    let material_library = self.parse_material_library()?;

    if material_library.is_some() {
      self.one_or_more_newlines()?;
    }

    let objects = self.parse_objects()?;

    self.zero_or_more_newlines();

    if let Some(s) = self.peek() {
      return self.error(format!("Expected end of input but got {}.", s));
    }

    Ok(ObjSet {
      material_library: material_library.map(|s| s.to_owned()),
      objects,
    })
  }

  fn parse_groups(&mut self) -> Result<Vec<GroupName>, ParseError> {
    let mut groups = Vec::new();

    loop {
      // ends the list of group names
      // g without any name is valid and means default group
      if let Some("\n") = self.peek() {
        break;
      }

      let name = self.parse_str()?;
      groups.push(name.to_owned());
    }

    Ok(groups)
  }

  fn parse_smoothing_groups(&mut self) -> Result<Vec<u32>, ParseError> {
    let mut groups = Vec::new();

    if self.try(|p| p.parse_tag("off")).is_none() {
      loop {
        let group = self.parse_u32()?;
        groups.push(group);

        if let Some("\n") = self.peek() {
          break;
        }
      }
    }

    Ok(groups)
  }
}

#[test]
fn test_parse() {
  use self::Primitive::{Line, Triangle};

  let test_case = r#"
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
f 45 41 44 48"#;

  let expected = Ok(ObjSet {
    material_library: Some("untitled.mtl".to_owned()),
    objects: vec![
      Object {
        name: "Cube.001".to_owned(),
        vertices: vec![
          (-1, -1, 1),
          (-1, -1, -1),
          (1, -1, -1),
          (1, -1, 1),
          (-1, 1, 1),
          (-1, 1, -1),
          (1, 1, -1),
          (1, 1, 1),
        ]
        .into_iter()
        .map(|(x, y, z)| Vertex {
          x: x as f64,
          y: y as f64,
          z: z as f64,
        })
        .collect(),
        tex_vertices: vec![],
        normals: vec![],
        geometry: vec![Geometry {
          material_name: Some("None".to_owned()),
          shapes: vec![
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
            (3, 0, 1),
            (3, 1, 2),
            (4, 7, 6),
            (4, 6, 5),
          ]
          .into_iter()
          .map(|(x, y, z)| Shape {
            primitive: Triangle((x, None, None), (y, None, None), (z, None, None)),
            groups: vec![],
            smoothing_groups: vec![],
          })
          .collect(),
        }],
      },
      Object {
        name: "Circle".to_owned(),
        vertices: vec![
          (0.0, 0.0, -1.0),
          (-0.19509, 0.0, -0.980785),
          (-0.382683, 0.0, -0.92388),
          (-0.55557, 0.0, -0.83147),
          (-0.707107, 0.0, -0.707107),
          (-0.83147, 0.0, -0.55557),
          (-0.92388, 0.0, -0.382683),
          (-0.980785, 0.0, -0.19509),
          (-1.0, 0.0, 0.0),
          (-0.980785, 0.0, 0.19509),
          (-0.92388, 0.0, 0.382683),
          (-0.83147, 0.0, 0.55557),
          (-0.707107, 0.0, 0.707107),
          (-0.55557, 0.0, 0.83147),
          (-0.382683, 0.0, 0.92388),
          (-0.19509, 0.0, 0.980785),
          (0.0, 0.0, 1.0),
          (0.195091, 0.0, 0.980785),
          (0.382684, 0.0, 0.923879),
          (0.555571, 0.0, 0.831469),
          (0.707107, 0.0, 0.707106),
          (0.83147, 0.0, 0.55557),
          (0.92388, 0.0, 0.382683),
          (0.980785, 0.0, 0.195089),
          (1.0, 0.0, -0.000001),
          (0.980785, 0.0, -0.195091),
          (0.923879, 0.0, -0.382684),
          (0.831469, 0.0, -0.555571),
          (0.707106, 0.0, -0.707108),
          (0.555569, 0.0, -0.83147),
          (0.382682, 0.0, -0.92388),
          (0.195089, 0.0, -0.980786),
        ]
        .into_iter()
        .map(|(x, y, z)| Vertex { x, y, z })
        .collect(),
        tex_vertices: vec![],
        normals: vec![],
        geometry: vec![Geometry {
          material_name: None,
          shapes: vec![
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 8),
            (10, 9),
            (11, 10),
            (12, 11),
            (13, 12),
            (14, 13),
            (15, 14),
            (16, 15),
            (17, 16),
            (18, 17),
            (19, 18),
            (20, 19),
            (21, 20),
            (22, 21),
            (23, 22),
            (24, 23),
            (25, 24),
            (26, 25),
            (27, 26),
            (28, 27),
            (29, 28),
            (30, 29),
            (31, 30),
            (0, 31),
          ]
          .into_iter()
          .map(|(x, y)| Shape {
            primitive: Line((x, None, None), (y, None, None)),
            groups: vec![],
            smoothing_groups: vec![],
          })
          .collect(),
        }],
      },
      Object {
        name: "Cube".to_owned(),
        vertices: vec![
          (1.0, -1.0, -1.0),
          (1.0, -1.0, 1.0),
          (-1.0, -1.0, 1.0),
          (-1.0, -1.0, -1.0),
          (1.0, 1.0, -0.999999),
          (0.999999, 1.0, 1.000001),
          (-1.0, 1.0, 1.0),
          (-1.0, 1.0, -1.0),
        ]
        .into_iter()
        .map(|(x, y, z)| Vertex { x, y, z })
        .collect(),
        tex_vertices: vec![],
        normals: vec![],
        geometry: vec![Geometry {
          material_name: Some("Material".to_owned()),
          shapes: vec![
            (3, 0, 1),
            (3, 1, 2),
            (5, 4, 7),
            (5, 7, 6),
            (1, 0, 4),
            (1, 4, 5),
            (2, 1, 5),
            (2, 5, 6),
            (3, 2, 6),
            (3, 6, 7),
            (7, 4, 0),
            (7, 0, 3),
          ]
          .into_iter()
          .map(|(x, y, z)| Shape {
            primitive: Triangle((x, None, None), (y, None, None), (z, None, None)),
            groups: vec![],
            smoothing_groups: vec![],
          })
          .collect(),
        }],
      },
    ],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_cube() {
  use self::Primitive::Triangle;

  let test_case = r#"
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
f 5/5 1/13 4/14 8/6"#;

  let expected = Ok(ObjSet {
    material_library: Some("cube.mtl".to_owned()),
    objects: vec![Object {
      name: "Cube".to_owned(),
      vertices: vec![
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex {
        x: x as f64,
        y: y as f64,
        z: z as f64,
      })
      .collect(),
      tex_vertices: vec![
        (1.004952, 0.498633),
        (0.754996, 0.498236),
        (0.755393, 0.248279),
        (1.005349, 0.248677),
        (0.255083, 0.497442),
        (0.25548, 0.247485),
        (0.505437, 0.247882),
        (0.505039, 0.497839),
        (0.754598, 0.748193),
        (0.504642, 0.747795),
        (0.505834, -0.002074),
        (0.75579, -0.001677),
        (0.005127, 0.497044),
        (0.005524, 0.247088),
      ]
      .into_iter()
      .map(|(u, v)| TVertex { u, v, w: 0. })
      .collect(),
      normals: vec![],
      geometry: vec![Geometry {
        material_name: Some("Material".to_owned()),
        shapes: vec![
          (3, 3, 0, 0, 1, 1),
          (3, 3, 1, 1, 2, 2),
          (5, 7, 4, 4, 7, 5),
          (5, 7, 7, 5, 6, 6),
          (1, 1, 0, 8, 4, 9),
          (1, 1, 4, 9, 5, 7),
          (2, 2, 1, 1, 5, 7),
          (2, 2, 5, 7, 6, 6),
          (3, 11, 2, 2, 6, 6),
          (3, 11, 6, 6, 7, 10),
          (7, 5, 4, 4, 0, 12),
          (7, 5, 0, 12, 3, 13),
        ]
        .into_iter()
        .map(|(vx, tx, vy, ty, vz, tz)| Shape {
          primitive: Triangle(
            (vx, Some(tx), None),
            (vy, Some(ty), None),
            (vz, Some(tz), None),
          ),
          groups: vec![],
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_cube_anonymous_object() {
  use self::Primitive::Triangle;

  let test_case = r#"
# Blender v2.71 (sub 0) OBJ File: 'cube.blend'
# www.blender.org
mtllib cube.mtl
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
f 5/5 1/13 4/14 8/6"#;

  let expected = Ok(ObjSet {
    material_library: Some("cube.mtl".to_owned()),
    objects: vec![Object {
      name: String::new(),
      vertices: vec![
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex {
        x: x as f64,
        y: y as f64,
        z: z as f64,
      })
      .collect(),
      tex_vertices: vec![
        (1.004952, 0.498633),
        (0.754996, 0.498236),
        (0.755393, 0.248279),
        (1.005349, 0.248677),
        (0.255083, 0.497442),
        (0.25548, 0.247485),
        (0.505437, 0.247882),
        (0.505039, 0.497839),
        (0.754598, 0.748193),
        (0.504642, 0.747795),
        (0.505834, -0.002074),
        (0.75579, -0.001677),
        (0.005127, 0.497044),
        (0.005524, 0.247088),
      ]
      .into_iter()
      .map(|(u, v)| TVertex { u, v, w: 0. })
      .collect(),
      normals: vec![],
      geometry: vec![Geometry {
        material_name: Some("Material".to_owned()),
        shapes: vec![
          (3, 3, 0, 0, 1, 1),
          (3, 3, 1, 1, 2, 2),
          (5, 7, 4, 4, 7, 5),
          (5, 7, 7, 5, 6, 6),
          (1, 1, 0, 8, 4, 9),
          (1, 1, 4, 9, 5, 7),
          (2, 2, 1, 1, 5, 7),
          (2, 2, 5, 7, 6, 6),
          (3, 11, 2, 2, 6, 6),
          (3, 11, 6, 6, 7, 10),
          (7, 5, 4, 4, 0, 12),
          (7, 5, 0, 12, 3, 13),
        ]
        .into_iter()
        .map(|(vx, tx, vy, ty, vz, tz)| Shape {
          primitive: Triangle(
            (vx, Some(tx), None),
            (vy, Some(ty), None),
            (vz, Some(tz), None),
          ),
          groups: vec![],
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_cube_tex_vert_missing_vw() {
  use self::Primitive::Triangle;

  let test_case = r#"
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
vt 1.004952
vt 0.754996
vt 0.755393
vt 1.005349
vt 0.255083
vt 0.255480
vt 0.505437
vt 0.505039
vt 0.754598
vt 0.504642
vt 0.505834
vt 0.755790
vt 0.005127
vt 0.005524
usemtl Material
s off
f 1/1 2/2 3/3 4/4
f 5/5 8/6 7/7 6/8
f 1/9 5/10 6/8 2/2
f 2/2 6/8 7/7 3/3
f 3/3 7/7 8/11 4/12
f 5/5 1/13 4/14 8/6"#;

  let expected = Ok(ObjSet {
    material_library: Some("cube.mtl".to_owned()),
    objects: vec![Object {
      name: "Cube".to_owned(),
      vertices: vec![
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex {
        x: x as f64,
        y: y as f64,
        z: z as f64,
      })
      .collect(),
      tex_vertices: vec![
        1.004952, 0.754996, 0.755393, 1.005349, 0.255083, 0.25548, 0.505437, 0.505039, 0.754598,
        0.504642, 0.505834, 0.75579, 0.005127, 0.005524,
      ]
      .into_iter()
      .map(|u| TVertex { u, v: 0., w: 0. })
      .collect(),
      normals: vec![],
      geometry: vec![Geometry {
        material_name: Some("Material".to_owned()),
        shapes: vec![
          (3, 3, 0, 0, 1, 1),
          (3, 3, 1, 1, 2, 2),
          (5, 7, 4, 4, 7, 5),
          (5, 7, 7, 5, 6, 6),
          (1, 1, 0, 8, 4, 9),
          (1, 1, 4, 9, 5, 7),
          (2, 2, 1, 1, 5, 7),
          (2, 2, 5, 7, 6, 6),
          (3, 11, 2, 2, 6, 6),
          (3, 11, 6, 6, 7, 10),
          (7, 5, 4, 4, 0, 12),
          (7, 5, 0, 12, 3, 13),
        ]
        .into_iter()
        .map(|(vx, tx, vy, ty, vz, tz)| Shape {
          primitive: Triangle(
            (vx, Some(tx), None),
            (vy, Some(ty), None),
            (vz, Some(tz), None),
          ),
          groups: vec![],
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_cube_3d_tex_vert() {
  use self::Primitive::Triangle;

  let test_case = r#"
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
vt 1.004952 0.498633 1.0
vt 0.754996 0.498236 1.0
vt 0.755393 0.248279 1.0
vt 1.005349 0.248677 1.0
vt 0.255083 0.497442 1.0
vt 0.255480 0.247485 1.0
vt 0.505437 0.247882 1.0
vt 0.505039 0.497839 1.0
vt 0.754598 0.748193 1.0
vt 0.504642 0.747795 1.0
vt 0.505834 -0.002074 1.0
vt 0.755790 -0.001677 1.0
vt 0.005127 0.497044 1.0
vt 0.005524 0.247088 1.0
usemtl Material
s off
f 1/1 2/2 3/3 4/4
f 5/5 8/6 7/7 6/8
f 1/9 5/10 6/8 2/2
f 2/2 6/8 7/7 3/3
f 3/3 7/7 8/11 4/12
f 5/5 1/13 4/14 8/6
"#;

  let expected = Ok(ObjSet {
    material_library: Some("cube.mtl".to_owned()),
    objects: vec![Object {
      name: "Cube".to_owned(),
      vertices: vec![
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex {
        x: x as f64,
        y: y as f64,
        z: z as f64,
      })
      .collect(),
      tex_vertices: vec![
        (1.004952, 0.498633),
        (0.754996, 0.498236),
        (0.755393, 0.248279),
        (1.005349, 0.248677),
        (0.255083, 0.497442),
        (0.25548, 0.247485),
        (0.505437, 0.247882),
        (0.505039, 0.497839),
        (0.754598, 0.748193),
        (0.504642, 0.747795),
        (0.505834, -0.002074),
        (0.75579, -0.001677),
        (0.005127, 0.497044),
        (0.005524, 0.247088),
      ]
      .into_iter()
      .map(|(u, v)| TVertex { u, v, w: 1. })
      .collect(),
      normals: vec![],
      geometry: vec![Geometry {
        material_name: Some("Material".to_owned()),
        shapes: vec![
          (3, 3, 0, 0, 1, 1),
          (3, 3, 1, 1, 2, 2),
          (5, 7, 4, 4, 7, 5),
          (5, 7, 7, 5, 6, 6),
          (1, 1, 0, 8, 4, 9),
          (1, 1, 4, 9, 5, 7),
          (2, 2, 1, 1, 5, 7),
          (2, 2, 5, 7, 6, 6),
          (3, 11, 2, 2, 6, 6),
          (3, 11, 6, 6, 7, 10),
          (7, 5, 4, 4, 0, 12),
          (7, 5, 0, 12, 3, 13),
        ]
        .into_iter()
        .map(|(vx, tx, vy, ty, vz, tz)| Shape {
          primitive: Triangle(
            (vx, Some(tx), None),
            (vy, Some(ty), None),
            (vz, Some(tz), None),
          ),
          groups: vec![],
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_normals_no_tex() {
  use self::Primitive::Triangle;

  let test_case = r#"
# Blender v2.70 (sub 4) OBJ File: ''
# www.blender.org
mtllib normal-cone.mtl
o Cone
v 0.000000  -1.000000 -1.000000
v 0.000000   1.000000  0.000000
v 0.195090  -1.000000 -0.980785
v 0.382683  -1.000000 -0.923880
v 0.555570  -1.000000 -0.831470
v 0.707107  -1.000000 -0.707107
v 0.831470  -1.000000 -0.555570
v 0.923880  -1.000000 -0.382683
v 0.980785  -1.000000 -0.195090
v 1.000000  -1.000000 -0.000000
v 0.980785  -1.000000  0.195090
v 0.923880  -1.000000  0.382683
v 0.831470  -1.000000  0.555570
v 0.707107  -1.000000  0.707107
v 0.555570  -1.000000  0.831470
v 0.382683  -1.000000  0.923880
v 0.195090  -1.000000  0.980785
v -0.000000 -1.000000  1.000000
v -0.195091 -1.000000  0.980785
v -0.382684 -1.000000  0.923879
v -0.555571 -1.000000  0.831469
v -0.707107 -1.000000  0.707106
v -0.831470 -1.000000  0.555570
v -0.923880 -1.000000  0.382683
v -0.980785 -1.000000  0.195089
v -1.000000 -1.000000 -0.000001
v -0.980785 -1.000000 -0.195091
v -0.923879 -1.000000 -0.382684
v -0.831469 -1.000000 -0.555571
v -0.707106 -1.000000 -0.707108
v -0.555569 -1.000000 -0.831470
v -0.382682 -1.000000 -0.923880
v -0.195089 -1.000000 -0.980786
vn -0.259887 0.445488 -0.856737
vn 0.087754 0.445488 -0.890977
vn -0.422035 0.445488 -0.789574
vn -0.567964 0.445488 -0.692068
vn -0.692066 0.445488 -0.567966
vn -0.789573 0.445488 -0.422037
vn -0.856737 0.445488 -0.259889
vn -0.890977 0.445488 -0.087754
vn -0.890977 0.445488 0.087753
vn -0.856737 0.445488 0.259887
vn -0.789574 0.445488 0.422035
vn -0.692067 0.445488 0.567964
vn -0.567965 0.445488 0.692066
vn -0.422036 0.445488 0.789573
vn -0.259889 0.445488 0.856737
vn -0.087754 0.445488 0.890977
vn 0.087753 0.445488 0.890977
vn 0.259888 0.445488 0.856737
vn 0.422036 0.445488 0.789573
vn 0.567965 0.445488 0.692067
vn 0.692067 0.445488 0.567965
vn 0.789573 0.445488 0.422035
vn 0.856737 0.445488 0.259888
vn 0.890977 0.445488 0.087753
vn 0.890977 0.445488 -0.087754
vn 0.856737 0.445488 -0.259888
vn 0.789573 0.445488 -0.422036
vn 0.692067 0.445488 -0.567965
vn 0.567965 0.445488 -0.692067
vn 0.422036 0.445488 -0.789573
vn -0.087753 0.445488 -0.890977
vn 0.259888 0.445488 -0.856737
vn 0.000000 -1.000000 -0.000000
usemtl Material.002
s off
f 32//1 2//1 33//1
f 1//2 2//2 3//2
f 31//3 2//3 32//3
f 30//4 2//4 31//4
f 29//5 2//5 30//5
f 28//6 2//6 29//6
f 27//7 2//7 28//7
f 26//8 2//8 27//8
f 25//9 2//9 26//9
f 24//10 2//10 25//10
f 23//11 2//11 24//11
f 22//12 2//12 23//12
f 21//13 2//13 22//13
f 20//14 2//14 21//14
f 19//15 2//15 20//15
f 18//16 2//16 19//16
f 17//17 2//17 18//17
f 16//18 2//18 17//18
f 15//19 2//19 16//19
f 14//20 2//20 15//20
f 13//21 2//21 14//21
f 12//22 2//22 13//22
f 11//23 2//23 12//23
f 10//24 2//24 11//24
f 9//25 2//25 10//25
f 8//26 2//26 9//26
f 7//27 2//27 8//27
f 6//28 2//28 7//28
f 5//29 2//29 6//29
f 4//30 2//30 5//30
f 33//31 2//31 1//31
f 3//32 2//32 4//32
"#;

  let expected = Ok(ObjSet {
    material_library: Some("normal-cone.mtl".to_owned()),
    objects: vec![Object {
      name: "Cone".to_owned(),
      vertices: vec![
        (0.000000, -1.000000, -1.000000),
        (0.000000, 1.000000, 0.000000),
        (0.195090, -1.000000, -0.980785),
        (0.382683, -1.000000, -0.923880),
        (0.555570, -1.000000, -0.831470),
        (0.707107, -1.000000, -0.707107),
        (0.831470, -1.000000, -0.555570),
        (0.923880, -1.000000, -0.382683),
        (0.980785, -1.000000, -0.195090),
        (1.000000, -1.000000, -0.000000),
        (0.980785, -1.000000, 0.195090),
        (0.923880, -1.000000, 0.382683),
        (0.831470, -1.000000, 0.555570),
        (0.707107, -1.000000, 0.707107),
        (0.555570, -1.000000, 0.831470),
        (0.382683, -1.000000, 0.923880),
        (0.195090, -1.000000, 0.980785),
        (-0.000000, -1.000000, 1.000000),
        (-0.195091, -1.000000, 0.980785),
        (-0.382684, -1.000000, 0.923879),
        (-0.555571, -1.000000, 0.831469),
        (-0.707107, -1.000000, 0.707106),
        (-0.831470, -1.000000, 0.555570),
        (-0.923880, -1.000000, 0.382683),
        (-0.980785, -1.000000, 0.195089),
        (-1.000000, -1.000000, -0.000001),
        (-0.980785, -1.000000, -0.195091),
        (-0.923879, -1.000000, -0.382684),
        (-0.831469, -1.000000, -0.555571),
        (-0.707106, -1.000000, -0.707108),
        (-0.555569, -1.000000, -0.831470),
        (-0.382682, -1.000000, -0.923880),
        (-0.195089, -1.000000, -0.980786),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex { x, y, z })
      .collect(),
      tex_vertices: vec![],
      normals: vec![
        (-0.259887, 0.445488, -0.856737),
        (0.087754, 0.445488, -0.890977),
        (-0.422035, 0.445488, -0.789574),
        (-0.567964, 0.445488, -0.692068),
        (-0.692066, 0.445488, -0.567966),
        (-0.789573, 0.445488, -0.422037),
        (-0.856737, 0.445488, -0.259889),
        (-0.890977, 0.445488, -0.087754),
        (-0.890977, 0.445488, 0.087753),
        (-0.856737, 0.445488, 0.259887),
        (-0.789574, 0.445488, 0.422035),
        (-0.692067, 0.445488, 0.567964),
        (-0.567965, 0.445488, 0.692066),
        (-0.422036, 0.445488, 0.789573),
        (-0.259889, 0.445488, 0.856737),
        (-0.087754, 0.445488, 0.890977),
        (0.087753, 0.445488, 0.890977),
        (0.259888, 0.445488, 0.856737),
        (0.422036, 0.445488, 0.789573),
        (0.567965, 0.445488, 0.692067),
        (0.692067, 0.445488, 0.567965),
        (0.789573, 0.445488, 0.422035),
        (0.856737, 0.445488, 0.259888),
        (0.890977, 0.445488, 0.087753),
        (0.890977, 0.445488, -0.087754),
        (0.856737, 0.445488, -0.259888),
        (0.789573, 0.445488, -0.422036),
        (0.692067, 0.445488, -0.567965),
        (0.567965, 0.445488, -0.692067),
        (0.422036, 0.445488, -0.789573),
        (-0.087753, 0.445488, -0.890977),
        (0.259888, 0.445488, -0.856737),
        (0.000000, -1.000000, -0.000000),
      ]
      .into_iter()
      .map(|(x, y, z)| Normal { x, y, z })
      .collect(),
      geometry: vec![Geometry {
        material_name: Some("Material.002".to_owned()),
        shapes: vec![
          (32, 0, 31, 0, 1, 0),
          (2, 1, 0, 1, 1, 1),
          (31, 2, 30, 2, 1, 2),
          (30, 3, 29, 3, 1, 3),
          (29, 4, 28, 4, 1, 4),
          (28, 5, 27, 5, 1, 5),
          (27, 6, 26, 6, 1, 6),
          (26, 7, 25, 7, 1, 7),
          (25, 8, 24, 8, 1, 8),
          (24, 9, 23, 9, 1, 9),
          (23, 10, 22, 10, 1, 10),
          (22, 11, 21, 11, 1, 11),
          (21, 12, 20, 12, 1, 12),
          (20, 13, 19, 13, 1, 13),
          (19, 14, 18, 14, 1, 14),
          (18, 15, 17, 15, 1, 15),
          (17, 16, 16, 16, 1, 16),
          (16, 17, 15, 17, 1, 17),
          (15, 18, 14, 18, 1, 18),
          (14, 19, 13, 19, 1, 19),
          (13, 20, 12, 20, 1, 20),
          (12, 21, 11, 21, 1, 21),
          (11, 22, 10, 22, 1, 22),
          (10, 23, 9, 23, 1, 23),
          (9, 24, 8, 24, 1, 24),
          (8, 25, 7, 25, 1, 25),
          (7, 26, 6, 26, 1, 26),
          (6, 27, 5, 27, 1, 27),
          (5, 28, 4, 28, 1, 28),
          (4, 29, 3, 29, 1, 29),
          (0, 30, 32, 30, 1, 30),
          (3, 31, 2, 31, 1, 31),
        ]
        .into_iter()
        .map(|(vx, nx, vy, ny, vz, nz)| Shape {
          primitive: Triangle(
            (vx, None, Some(nx)),
            (vy, None, Some(ny)),
            (vz, None, Some(nz)),
          ),
          groups: vec![],
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  });
  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_smoothing_groups() {
  use self::Primitive::Triangle;

  let test_case = r#"
# Blender v2.72 (sub 0) OBJ File: 'dome.blend'
# www.blender.org
mtllib dome.mtl
o Dome
v -0.382683 0.923880 0.000000
v -0.707107 0.707107 0.000000
v -0.923880 0.382683 0.000000
v -1.000000 -0.000000 0.000000
v -0.270598 0.923880 -0.270598
v -0.500000 0.707107 -0.500000
v -0.653282 0.382683 -0.653281
v -0.707107 -0.000000 -0.707107
v -0.000000 0.923880 -0.382683
v -0.000000 0.707107 -0.707107
v -0.000000 0.382683 -0.923879
v -0.000000 -0.000000 -1.000000
v -0.000000 1.000000 0.000000
v 0.270598 0.923880 -0.270598
v 0.500000 0.707107 -0.500000
v 0.653281 0.382683 -0.653281
v 0.707106 -0.000000 -0.707107
v 0.382683 0.923880 -0.000000
v 0.707106 0.707107 -0.000000
v 0.923879 0.382683 -0.000000
v 1.000000 -0.000000 -0.000000
v 0.270598 0.923880 0.270598
v 0.500000 0.707107 0.500000
v 0.653281 0.382683 0.653281
v 0.707106 -0.000000 0.707107
v -0.000000 0.923880 0.382683
v -0.000000 0.707107 0.707107
v -0.000000 0.382683 0.923879
v -0.000000 -0.000000 1.000000
v -0.270598 0.923880 0.270598
v -0.500000 0.707107 0.500000
v -0.653281 0.382683 0.653281
v -0.707107 -0.000000 0.707107
usemtl None
s 1
f 4 3 7
f 3 2 6
f 1 5 6
f 7 11 12
f 6 10 11
f 5 9 10
f 11 16 17
f 11 10 15
f 10 9 14
f 16 20 21
f 15 19 20
f 14 18 19
f 20 24 25
f 20 19 23
f 18 22 23
f 24 28 29
f 24 23 27
f 23 22 26
f 28 32 33
f 28 27 31
f 27 26 30
f 1 13 5
f 5 13 9
f 9 13 14
f 14 13 18
f 18 13 22
f 22 13 26
f 26 13 30
f 32 3 4
f 31 2 3
f 30 1 2
f 30 13 1
f 8 4 7
f 7 3 6
f 2 1 6
f 8 7 12
f 7 6 11
f 6 5 10
f 12 11 17
f 16 11 15
f 15 10 14
f 17 16 21
f 16 15 20
f 15 14 19
f 21 20 25
f 24 20 23
f 19 18 23
f 25 24 29
f 28 24 27
f 27 23 26
f 29 28 33
f 32 28 31
f 31 27 30
f 33 32 4
f 32 31 3
f 31 30 2
s 2
f 33 4 8
f 29 33 25
f 12 17 21
f 12 33 8
f 33 21 25
f 21 33 12
"#;

  let expected = Ok(ObjSet {
    material_library: Some("dome.mtl".to_owned()),
    objects: vec![Object {
      name: "Dome".to_owned(),
      vertices: vec![
        (-0.382683, 0.92388, 0.0),
        (-0.707107, 0.707107, 0.0),
        (-0.92388, 0.382683, 0.0),
        (-1.0, 0.0, 0.0),
        (-0.270598, 0.92388, -0.270598),
        (-0.5, 0.707107, -0.5),
        (-0.653282, 0.382683, -0.653281),
        (-0.707107, 0.0, -0.707107),
        (0.0, 0.92388, -0.382683),
        (0.0, 0.707107, -0.707107),
        (0.0, 0.382683, -0.923879),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
        (0.270598, 0.92388, -0.270598),
        (0.5, 0.707107, -0.5),
        (0.653281, 0.382683, -0.653281),
        (0.707106, 0.0, -0.707107),
        (0.382683, 0.92388, 0.0),
        (0.707106, 0.707107, 0.0),
        (0.923879, 0.382683, 0.0),
        (1.0, 0.0, 0.0),
        (0.270598, 0.92388, 0.270598),
        (0.5, 0.707107, 0.5),
        (0.653281, 0.382683, 0.653281),
        (0.707106, 0.0, 0.707107),
        (0.0, 0.92388, 0.382683),
        (0.0, 0.707107, 0.707107),
        (0.0, 0.382683, 0.923879),
        (0.0, 0.0, 1.0),
        (-0.270598, 0.92388, 0.270598),
        (-0.5, 0.707107, 0.5),
        (-0.653281, 0.382683, 0.653281),
        (-0.707107, 0.0, 0.707107),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex { x, y, z })
      .collect(),
      tex_vertices: vec![],
      normals: vec![],
      geometry: vec![Geometry {
        material_name: Some("None".to_owned()),
        shapes: vec![
          (6, 3, 2, 1),
          (5, 2, 1, 1),
          (5, 0, 4, 1),
          (11, 6, 10, 1),
          (10, 5, 9, 1),
          (9, 4, 8, 1),
          (16, 10, 15, 1),
          (14, 10, 9, 1),
          (13, 9, 8, 1),
          (20, 15, 19, 1),
          (19, 14, 18, 1),
          (18, 13, 17, 1),
          (24, 19, 23, 1),
          (22, 19, 18, 1),
          (22, 17, 21, 1),
          (28, 23, 27, 1),
          (26, 23, 22, 1),
          (25, 22, 21, 1),
          (32, 27, 31, 1),
          (30, 27, 26, 1),
          (29, 26, 25, 1),
          (4, 0, 12, 1),
          (8, 4, 12, 1),
          (13, 8, 12, 1),
          (17, 13, 12, 1),
          (21, 17, 12, 1),
          (25, 21, 12, 1),
          (29, 25, 12, 1),
          (3, 31, 2, 1),
          (2, 30, 1, 1),
          (1, 29, 0, 1),
          (0, 29, 12, 1),
          (6, 7, 3, 1),
          (5, 6, 2, 1),
          (5, 1, 0, 1),
          (11, 7, 6, 1),
          (10, 6, 5, 1),
          (9, 5, 4, 1),
          (16, 11, 10, 1),
          (14, 15, 10, 1),
          (13, 14, 9, 1),
          (20, 16, 15, 1),
          (19, 15, 14, 1),
          (18, 14, 13, 1),
          (24, 20, 19, 1),
          (22, 23, 19, 1),
          (22, 18, 17, 1),
          (28, 24, 23, 1),
          (26, 27, 23, 1),
          (25, 26, 22, 1),
          (32, 28, 27, 1),
          (30, 31, 27, 1),
          (29, 30, 26, 1),
          (3, 32, 31, 1),
          (2, 31, 30, 1),
          (1, 30, 29, 1),
          (7, 32, 3, 2),
          (24, 28, 32, 2),
          (20, 11, 16, 2),
          (7, 11, 32, 2),
          (24, 32, 20, 2),
          (11, 20, 32, 2),
        ]
        .into_iter()
        .map(|(x, y, z, s)| Shape {
          primitive: Triangle((x, None, None), (y, None, None), (z, None, None)),
          groups: vec![],
          smoothing_groups: vec![s],
        })
        .collect(),
      }],
    }],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn no_mtls() {
  use self::Primitive::Triangle;

  let test_case = r#"
# Blender v2.71 (sub 0) OBJ File: 'cube.blend'
# www.blender.org
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
s off
f 1/1 2/2 3/3 4/4
f 5/5 8/6 7/7 6/8
f 1/9 5/10 6/8 2/2
f 2/2 6/8 7/7 3/3
f 3/3 7/7 8/11 4/12
f 5/5 1/13 4/14 8/6
"#;

  let expected = Ok(ObjSet {
    material_library: None,
    objects: vec![Object {
      name: "Cube".to_owned(),
      vertices: vec![
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex { x, y, z })
      .collect(),
      tex_vertices: vec![
        (1.004952, 0.498633),
        (0.754996, 0.498236),
        (0.755393, 0.248279),
        (1.005349, 0.248677),
        (0.255083, 0.497442),
        (0.25548, 0.247485),
        (0.505437, 0.247882),
        (0.505039, 0.497839),
        (0.754598, 0.748193),
        (0.504642, 0.747795),
        (0.505834, -0.002074),
        (0.75579, -0.001677),
        (0.005127, 0.497044),
        (0.005524, 0.247088),
      ]
      .into_iter()
      .map(|(u, v)| TVertex { u, v, w: 0. })
      .collect(),
      normals: vec![],
      geometry: vec![Geometry {
        material_name: None,
        shapes: vec![
          (3, 3, 0, 0, 1, 1),
          (3, 3, 1, 1, 2, 2),
          (5, 7, 4, 4, 7, 5),
          (5, 7, 7, 5, 6, 6),
          (1, 1, 0, 8, 4, 9),
          (1, 1, 4, 9, 5, 7),
          (2, 2, 1, 1, 5, 7),
          (2, 2, 5, 7, 6, 6),
          (3, 11, 2, 2, 6, 6),
          (3, 11, 6, 6, 7, 10),
          (7, 5, 4, 4, 0, 12),
          (7, 5, 0, 12, 3, 13),
        ]
        .into_iter()
        .map(|(vx, tx, vy, ty, vz, tz)| Shape {
          primitive: Triangle(
            (vx, Some(tx), None),
            (vy, Some(ty), None),
            (vz, Some(tz), None),
          ),
          groups: vec![],
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn one_group() {
  use self::Primitive::Triangle;

  let input = "
o Cube
v 1.000000 -1.000000 -1.000000\r
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
g all
s off
f 1/1 2/2 3/3 4/4
f 5/5 8/6 7/7 6/8
f 1/9 5/10 6/8 2/2
f 2/2 6/8 7/7 3/3
f 3/3 7/7 8/11 4/12
f 5/5 1/13 4/14 8/6
";

  let expected = ObjSet {
    material_library: None,
    objects: vec![Object {
      name: "Cube".to_owned(),
      vertices: vec![
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
      ]
      .into_iter()
      .map(|(x, y, z)| Vertex { x, y, z })
      .collect(),
      tex_vertices: vec![
        (1.004952, 0.498633),
        (0.754996, 0.498236),
        (0.755393, 0.248279),
        (1.005349, 0.248677),
        (0.255083, 0.497442),
        (0.25548, 0.247485),
        (0.505437, 0.247882),
        (0.505039, 0.497839),
        (0.754598, 0.748193),
        (0.504642, 0.747795),
        (0.505834, -0.002074),
        (0.75579, -0.001677),
        (0.005127, 0.497044),
        (0.005524, 0.247088),
      ]
      .into_iter()
      .map(|(u, v)| TVertex { u, v, w: 0. })
      .collect(),
      normals: vec![],
      geometry: vec![Geometry {
        material_name: None,
        shapes: vec![
          (3, 3, 0, 0, 1, 1, "all"),
          (3, 3, 1, 1, 2, 2, "all"),
          (5, 7, 4, 4, 7, 5, "all"),
          (5, 7, 7, 5, 6, 6, "all"),
          (1, 1, 0, 8, 4, 9, "all"),
          (1, 1, 4, 9, 5, 7, "all"),
          (2, 2, 1, 1, 5, 7, "all"),
          (2, 2, 5, 7, 6, 6, "all"),
          (3, 11, 2, 2, 6, 6, "all"),
          (3, 11, 6, 6, 7, 10, "all"),
          (7, 5, 4, 4, 0, 12, "all"),
          (7, 5, 0, 12, 3, 13, "all"),
        ]
        .into_iter()
        .map(|(xv, xt, yv, yt, zv, zt, group)| Shape {
          primitive: Triangle(
            (xv, Some(xt), None),
            (yv, Some(yt), None),
            (zv, Some(zt), None),
          ),
          groups: vec![group.into()],
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  };

  assert_eq!(parse(input), Ok(expected));
}

#[test]
fn issue_54() {
  let input = include_str!("issue_54.obj");
  let _ = parse(input);
}

#[test]
fn several_groups() {
  use self::Primitive::Triangle;

  let input = r#"
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
s off
g face one
f 1/1 2/2 3/3 4/4
g face two
f 5/5 8/6 7/7 6/8
g face three
f 1/9 5/10 6/8 2/2
g face four
f 2/2 6/8 7/7 3/3
g face five
f 3/3 7/7 8/11 4/12
g face six
f 5/5 1/13 4/14 8/6"#;

  let expected = ObjSet {
    material_library: None,
    objects: vec![Object {
      name: "Cube".to_owned(),
      vertices: vec![
        Vertex {
          x: 1.0,
          y: -1.0,
          z: -1.0,
        },
        Vertex {
          x: 1.0,
          y: -1.0,
          z: 1.0,
        },
        Vertex {
          x: -1.0,
          y: -1.0,
          z: 1.0,
        },
        Vertex {
          x: -1.0,
          y: -1.0,
          z: -1.0,
        },
        Vertex {
          x: 1.0,
          y: 1.0,
          z: -1.0,
        },
        Vertex {
          x: 1.0,
          y: 1.0,
          z: 1.0,
        },
        Vertex {
          x: -1.0,
          y: 1.0,
          z: 1.0,
        },
        Vertex {
          x: -1.0,
          y: 1.0,
          z: -1.0,
        },
      ],
      tex_vertices: vec![
        TVertex {
          u: 1.004952,
          v: 0.498633,
          w: 0.,
        },
        TVertex {
          u: 0.754996,
          v: 0.498236,
          w: 0.,
        },
        TVertex {
          u: 0.755393,
          v: 0.248279,
          w: 0.,
        },
        TVertex {
          u: 1.005349,
          v: 0.248677,
          w: 0.,
        },
        TVertex {
          u: 0.255083,
          v: 0.497442,
          w: 0.,
        },
        TVertex {
          u: 0.25548,
          v: 0.247485,
          w: 0.,
        },
        TVertex {
          u: 0.505437,
          v: 0.247882,
          w: 0.,
        },
        TVertex {
          u: 0.505039,
          v: 0.497839,
          w: 0.,
        },
        TVertex {
          u: 0.754598,
          v: 0.748193,
          w: 0.,
        },
        TVertex {
          u: 0.504642,
          v: 0.747795,
          w: 0.,
        },
        TVertex {
          u: 0.505834,
          v: -0.002074,
          w: 0.,
        },
        TVertex {
          u: 0.75579,
          v: -0.001677,
          w: 0.,
        },
        TVertex {
          u: 0.005127,
          v: 0.497044,
          w: 0.,
        },
        TVertex {
          u: 0.005524,
          v: 0.247088,
          w: 0.,
        },
      ],
      normals: vec![],
      geometry: vec![Geometry {
        material_name: None,
        shapes: vec![
          (3, 3, 0, 0, 1, 1, vec!["face", "one"]),
          (3, 3, 1, 1, 2, 2, vec!["face", "one"]),
          (5, 7, 4, 4, 7, 5, vec!["face", "two"]),
          (5, 7, 7, 5, 6, 6, vec!["face", "two"]),
          (1, 1, 0, 8, 4, 9, vec!["face", "three"]),
          (1, 1, 4, 9, 5, 7, vec!["face", "three"]),
          (2, 2, 1, 1, 5, 7, vec!["face", "four"]),
          (2, 2, 5, 7, 6, 6, vec!["face", "four"]),
          (3, 11, 2, 2, 6, 6, vec!["face", "five"]),
          (3, 11, 6, 6, 7, 10, vec!["face", "five"]),
          (7, 5, 4, 4, 0, 12, vec!["face", "six"]),
          (7, 5, 0, 12, 3, 13, vec!["face", "six"]),
        ]
        .into_iter()
        .map(|(vx, tx, vy, ty, vz, tz, groups)| Shape {
          primitive: Triangle(
            (vx, Some(tx), None),
            (vy, Some(ty), None),
            (vz, Some(tz), None),
          ),
          groups: groups.into_iter().map(|s| s.into()).collect(),
          smoothing_groups: vec![],
        })
        .collect(),
      }],
    }],
  };

  assert_eq!(parse(input), Ok(expected));
}

/// Parses a wavefront `.obj` file, returning either the successfully parsed
/// file, or an error. Support in this parser for the full file format is
/// best-effort and realistically I will only end up supporting the subset
/// of the file format which falls under the "things I see exported from blender"
/// category.
pub fn parse<S: AsRef<str>>(input: S) -> Result<ObjSet, ParseError> {
  Parser::new(input.as_ref()).parse_objset()
}
