//! A parser for Wavefront's `.mtl` file format, for storing information about
//! the material of which a 3D mesh is composed.
use std::iter;
use std::result::{Result,Err,Ok};
pub use lex::ParseError;
use lex::Lexer;

/// A set of materials in one `.mtl` file.
#[deriving(Clone, Show, PartialEq)]
#[allow(missing_doc)]
pub struct MtlSet {
  pub materials: Vec<Material>,
}

/// A single material that can be applied to any face. They are generally
/// applied by using the Phong shading model.
#[deriving(Clone, Show)]
#[allow(missing_doc)]
pub struct Material {
  pub name: String,
  pub specular_coefficient: f64,
  pub color_ambient:  Color,
  pub color_diffuse:  Color,
  pub color_specular: Color,
  pub optical_density: Option<f64>,
  pub alpha: f64,
  pub illumination: Illumination,
  pub uv_map: Option<String>,
}

/// How a given material is supposed to be illuminated.
#[deriving(Clone, Copy, Show, Eq, PartialEq, Ord, PartialOrd)]
#[allow(missing_doc)]
pub enum Illumination {
  Ambient,
  AmbientDiffuse,
  AmbientDiffuseSpecular,
}

#[deriving(Clone, Copy, Show)]
#[allow(missing_doc)]
pub struct Color {
  pub r: f64,
  pub g: f64,
  pub b: f64,
}

fn fuzzy_cmp(x: f64, y: f64, delta: f64) -> Ordering {
  if (x - y).abs() <= delta {
    Equal
  } else if x < y {
    Less
  } else {
    Greater
  }
}

fn fuzzy_opt_cmp(x: Option<f64>, y: Option<f64>, delta: f64) -> Ordering {
  match (x, y) {
    (None, None) => Equal,
    (Some(_), None) => Greater,
    (None, Some(_)) => Less,
    (Some(x), Some(y)) => fuzzy_cmp(x, y, delta),
  }
}

impl PartialEq for Color {
  fn eq(&self, other: &Color) -> bool {
    self.partial_cmp(other).unwrap() == Equal
  }
}

impl PartialOrd for Color {
  fn partial_cmp(&self, other: &Color) -> Option<Ordering> {
    Some(fuzzy_cmp(self.r, other.r, 0.00001)
      .cmp(&fuzzy_cmp(self.g, other.g, 0.00001))
      .cmp(&fuzzy_cmp(self.b, other.b, 0.00001)))
  }
}

impl PartialEq for Material {
  fn eq(&self, other: &Material) -> bool {
    self.partial_cmp(other).unwrap() == Equal
  }
}

impl PartialOrd for Material {
  fn partial_cmp(&self, other: &Material) -> Option<Ordering> {
    Some(self.name.cmp(&other.name)
      .cmp(&fuzzy_cmp(self.specular_coefficient, other.specular_coefficient, 0.00001))
      .cmp(&self.color_ambient.partial_cmp(&other.color_ambient).unwrap())
      .cmp(&self.color_diffuse.partial_cmp(&other.color_diffuse).unwrap())
      .cmp(&self.color_specular.partial_cmp(&other.color_specular).unwrap())
      .cmp(&fuzzy_opt_cmp(self.optical_density, other.optical_density, 0.00001))
      .cmp(&fuzzy_cmp(self.alpha, other.alpha, 0.00001))
      .cmp(&self.illumination.cmp(&other.illumination))
      .cmp(&self.uv_map.cmp(&other.uv_map))) 
  }
}


/// Slices the underlying string in an option.
fn sliced<'a>(s: &'a Option<String>) -> Option<&'a str> {
  match *s {
    None => None,
    Some(ref s) => Some(s.as_slice()),
  }
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

  fn parse_newmtl(&mut self) -> Result<String, ParseError> {
    match sliced(&self.next()) {
      None => return self.error("Expected `newmtl` but got end of input.".into_string()),
      Some("newmtl") => {},
      Some(s) => return self.error(format!("Expected `newmtl` but got {}.", s)),
    }

    match self.next() {
      None => return self.error("Expected material name but got end of input.".into_string()),
      Some(s) => Ok(s),
    }
  }

  fn parse_f64(&mut self) -> Result<f64, ParseError> {
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

  fn parse_uint(&mut self) -> Result<uint, ParseError> {
    match sliced(&self.next()) {
      None =>
        return self.error("Expected uint but got end of input.".into_string()),
      Some(s) => {
        match from_str::<uint>(s) {
          None =>
            return self.error(format!("Expected uint but got {}.", s)),
          Some(ret) =>
            Ok(ret)
        }
      }
    }
  }

  fn parse_tag(&mut self, tag: &str) -> Result<(), ParseError> {
    match sliced(&self.next()) {
      None => return self.error(format!("Expected `{}` but got end of input.", tag)),
      Some(s) => {
        if s != tag {
          return self.error(format!("Expected `{}` but got {}.", tag, s));
        }
      }
    }

    Ok(())
  }

  fn parse_color(&mut self) -> Result<Color, ParseError> {
    let r = try!(self.parse_f64());
    let g = try!(self.parse_f64());
    let b = try!(self.parse_f64());

    Ok(Color { r: r, g: g, b: b })
  }

  fn parse_specular_coefficeint(&mut self) -> Result<f64, ParseError> {
    try!(self.parse_tag("Ns"));
    self.parse_f64()
  }

  fn parse_ambient_color(&mut self) -> Result<Color, ParseError> {
    try!(self.parse_tag("Ka"));
    self.parse_color()
  }

  fn parse_diffuse_color(&mut self) -> Result<Color, ParseError> {
    try!(self.parse_tag("Kd"));
    self.parse_color()
  }

  fn parse_specular_color(&mut self) -> Result<Color, ParseError> {
    try!(self.parse_tag("Ks"));
    self.parse_color()
  }

  fn parse_optical_density(&mut self) -> Result<Option<f64>, ParseError> {
    match sliced(&self.peek()) {
      Some("Ni") => {},
      _ => return Ok(None),
    }

    try!(self.parse_tag("Ni"));
    let optical_density = try!(self.parse_f64());
    Ok(Some(optical_density))
  }

  fn parse_dissolve(&mut self) -> Result<f64, ParseError> {
    try!(self.parse_tag("d"));
    self.parse_f64()
  }

  fn parse_illumination(&mut self) -> Result<Illumination, ParseError> {
    try!(self.parse_tag("illum"));
    match try!(self.parse_uint()) {
      0 => Ok(Ambient),
      1 => Ok(AmbientDiffuse),
      2 => Ok(AmbientDiffuseSpecular),
      n => self.error(format!("Unknown illumination model: {}.", n))
    }
  }

  fn parse_uv_map(&mut self) -> Result<Option<String>, ParseError> {
    match sliced(&self.peek()) {
      Some("map_Kd") => {},
      _ => return Ok(None),
    }

    try!(self.parse_tag("map_Kd"));
    match self.next() {
      None =>
        self.error("Expected texture path but got end of input.".into_string()),
      Some(s) =>
        Ok(Some(s))
    }
  }

  fn parse_material(&mut self) -> Result<Material, ParseError> {
    let name = try!(self.parse_newmtl());
    try!(self.one_or_more_newlines());
    let spec_coeff = try!(self.parse_specular_coefficeint());
    try!(self.one_or_more_newlines());
    let amb = try!(self.parse_ambient_color());
    try!(self.one_or_more_newlines());
    let diff = try!(self.parse_diffuse_color());
    try!(self.one_or_more_newlines());
    let spec = try!(self.parse_specular_color());
    try!(self.one_or_more_newlines());
    let optical_density = try!(self.parse_optical_density());
    if optical_density.is_some() { try!(self.one_or_more_newlines()); }
    let dissolve = try!(self.parse_dissolve());
    try!(self.one_or_more_newlines());
    let illum = try!(self.parse_illumination());
    try!(self.one_or_more_newlines());
    let uv_map = try!(self.parse_uv_map());
    if uv_map.is_some() { try!(self.one_or_more_newlines()); }

    Ok(Material {
      name: name,
      specular_coefficient: spec_coeff,
      color_ambient:  amb,
      color_diffuse:  diff,
      color_specular: spec,
      optical_density: optical_density,
      alpha: dissolve,
      illumination: illum,
      uv_map: uv_map,
    })
  }

  fn parse_mtlset(&mut self) -> Result<MtlSet, ParseError> {
    self.zero_or_more_newlines();

    let mut ret = Vec::new();

    loop {
      match sliced(&self.peek()) {
        Some("newmtl") => {
          ret.push(try!(self.parse_material()));
        },
        _ => break,
      }
    }

    Ok(MtlSet {
      materials: ret
    })
  }
}

/// Parses a wavefront `.mtl` file, returning either the successfully parsed
/// file, or an error. Support in this parser for the full file format is
/// best-effort and realistically I will only end up supporting the subset
/// of the file format which falls under the "shit I see exported from blender"
/// category.
pub fn parse(mut input: String) -> Result<MtlSet, ParseError> {
  input.push_str("\n");
  Parser::new(input.as_slice()).parse_mtlset()
}

#[test]
fn test_parse() {
  let test_case =
r#"
# Blender MTL File: 'None'
# Material Count: 2

# name
newmtl Material
# Phong specular coefficient
Ns 96.078431
# ambient color (weighted)
Ka 0.000000 0.000000 0.000000
# diffuse color (weighted)
Kd 0.640000 0.640000 0.640000
# dissolve factor (weighted)
Ks 0.500000 0.500000 0.500000
# optical density (refraction)
Ni 1.000000
# alpha
d 1.000000
# illumination: 0=ambient, 1=ambient+diffuse, 2=ambient+diffuse+specular
illum 2

newmtl None
Ns 0
# ambient
Ka 0.000000 0.000000 0.000000
# diffuse
Kd 0.8 0.8 0.8
# specular
Ks 0.8 0.8 0.8
d 1
illum 2

"#;

  let expected =
    Ok(MtlSet {
      materials: vec!(
        Material {
          name: "Material".into_string(),
          specular_coefficient: 96.078431,
          color_ambient:  Color { r: 0.0,  g: 0.0,  b: 0.0  },
          color_diffuse:  Color { r: 0.64, g: 0.64, b: 0.64 },
          color_specular: Color { r: 0.5,  g: 0.5,  b: 0.5  },
          optical_density: Some(1.0),
          alpha: 1.0,
          illumination: AmbientDiffuseSpecular,
          uv_map: None,
        },
        Material {
          name: "None".into_string(),
          specular_coefficient: 0.0,
          color_ambient:  Color { r: 0.0, g: 0.0, b: 0.0 },
          color_diffuse:  Color { r: 0.8, g: 0.8, b: 0.8 },
          color_specular: Color { r: 0.8, g: 0.8, b: 0.8 },
          optical_density: None,
          alpha: 1.0,
          illumination: AmbientDiffuseSpecular,
          uv_map: None,
        }
      )
    });

  assert_eq!(parse(test_case.into_string()), expected);
}

#[test]
fn test_cube() {
  let test_case =
r#"
# Blender MTL File: 'cube.blend'
# Material Count: 1

newmtl Material
Ns 96.078431
Ka 0.000000 0.000000 0.000000
Kd 0.640000 0.640000 0.640000
Ks 0.500000 0.500000 0.500000
Ni 1.000000
d 1.000000
illum 2
map_Kd cube-uv-num.png
"#;

  let expected =
    Ok(MtlSet {
      materials: vec!(
        Material {
          name: "Material".into_string(),
          specular_coefficient: 96.078431,
          color_ambient:  Color { r: 0.0,  g: 0.0,  b: 0.0  },
          color_diffuse:  Color { r: 0.64, g: 0.64, b: 0.64 },
          color_specular: Color { r: 0.5,  g: 0.5,  b: 0.5  },
          optical_density: Some(1.0),
          alpha: 1.0,
          illumination: AmbientDiffuseSpecular,
          uv_map: Some("cube-uv-num.png".into_string()),
        })
    });

  assert_eq!(parse(test_case.into_string()), expected);
}
