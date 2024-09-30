//! A parser for Wavefront's `.mtl` file format, for storing information about
//! the material of which a 3D mesh is composed.
use std::borrow::ToOwned;
use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};

pub use lex::ParseError;
use lex::{Lexer, PeekableLexer};
use util::OrderingExt;

/// A set of materials in one `.mtl` file.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct MtlSet {
  pub materials: Vec<Material>,
}

/// A single material that can be applied to any face. They are generally
/// applied by using the Phong shading model.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct Material {
  pub name: String,
  pub specular_coefficient: f64,
  pub color_ambient: Color,
  pub color_diffuse: Color,
  pub color_specular: Color,
  pub color_emissive: Option<Color>,
  pub optical_density: Option<f64>,
  pub alpha: f64,
  pub illumination: Illumination,
  pub ambient_map: Option<String>,
  pub diffuse_map: Option<String>,
  pub specular_map: Option<String>,
  pub specular_exponent_map: Option<String>,
  pub dissolve_map: Option<String>,
  pub displacement_map: Option<String>,
  pub decal_map: Option<String>,
  pub bump_map: Option<String>,
}

/// How a given material is supposed to be illuminated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[allow(missing_docs)]
pub enum Illumination {
  Ambient,
  AmbientDiffuse,
  AmbientDiffuseSpecular,
  ReflectionRayTrace,
}

#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub struct Color {
  pub r: f64,
  pub g: f64,
  pub b: f64,
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

fn fuzzy_opt_cmp(a: Option<f64>, b: Option<f64>, delta: f64) -> Ordering {
  match (a, b) {
    (None, None) => Equal,
    (Some(_), None) => Greater,
    (None, Some(_)) => Less,
    (Some(a), Some(b)) => fuzzy_cmp(a, b, delta),
  }
}

impl PartialEq for Color {
  fn eq(&self, other: &Color) -> bool {
    self.partial_cmp(other).unwrap() == Equal
  }
}

impl PartialOrd for Color {
  fn partial_cmp(&self, other: &Color) -> Option<Ordering> {
    Some(
      fuzzy_cmp(self.r, other.r, 0.00001)
        .lexico(|| fuzzy_cmp(self.g, other.g, 0.00001))
        .lexico(|| fuzzy_cmp(self.b, other.b, 0.00001)),
    )
  }
}

impl PartialEq for Material {
  fn eq(&self, other: &Material) -> bool {
    self.partial_cmp(other).unwrap() == Equal
  }
}

impl PartialOrd for Material {
  fn partial_cmp(&self, other: &Material) -> Option<Ordering> {
    Some(
      self
        .name
        .cmp(&other.name)
        .lexico(|| {
          fuzzy_cmp(
            self.specular_coefficient,
            other.specular_coefficient,
            0.00001,
          )
        })
        .lexico(|| {
          self
            .color_ambient
            .partial_cmp(&other.color_ambient)
            .unwrap()
        })
        .lexico(|| {
          self
            .color_diffuse
            .partial_cmp(&other.color_diffuse)
            .unwrap()
        })
        .lexico(|| {
          self
            .color_specular
            .partial_cmp(&other.color_specular)
            .unwrap()
        })
        .lexico(|| fuzzy_opt_cmp(self.optical_density, other.optical_density, 0.00001))
        .lexico(|| fuzzy_cmp(self.alpha, other.alpha, 0.00001))
        .lexico(|| self.illumination.cmp(&other.illumination))
        .lexico(|| self.ambient_map.cmp(&other.ambient_map))
        .lexico(|| self.diffuse_map.cmp(&other.diffuse_map))
        .lexico(|| self.specular_map.cmp(&other.specular_map))
        .lexico(|| self.specular_exponent_map.cmp(&other.specular_exponent_map))
        .lexico(|| self.dissolve_map.cmp(&other.dissolve_map))
        .lexico(|| self.displacement_map.cmp(&other.displacement_map))
        .lexico(|| self.decal_map.cmp(&other.decal_map))
        .lexico(|| self.bump_map.cmp(&other.bump_map)),
    )
  }
}

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

  fn error<A>(&self, msg: String) -> Result<A, ParseError> {
    Err(self.error_raw(msg))
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

  /// Possibly skips over some newlines.
  fn zero_or_more_newlines(&mut self) {
    while let Some("\n") = self.peek() {
      self.advance();
    }
  }
  /// Skips over some newlines, failing if it didn't manage to skip any.
  fn one_or_more_newlines(&mut self) -> Result<(), ParseError> {
    match self.peek() {
      None => {} // allow eof
      Some("\n") => {}
      Some(s) => return self.error(format!("Expected newline but got {}", s)),
    }

    self.zero_or_more_newlines();

    Ok(())
  }

  fn parse_newmtl(&mut self) -> Result<&'a str, ParseError> {
    match self.next() {
      None => return self.error("Expected `newmtl` but got end of input.".to_owned()),
      Some("newmtl") => {}
      Some(s) => return self.error(format!("Expected `newmtl` but got {}.", s)),
    }

    match self.next() {
      None => return self.error("Expected material name but got end of input.".to_owned()),
      Some(s) => Ok(s),
    }
  }

  fn parse_f64(&mut self) -> Result<f64, ParseError> {
    match self.next() {
      None => self.error("Expected f64 but got end of input.".to_owned()),
      Some(s) => s
        .parse()
        .map_err(|_| self.error_raw(format!("Expected f64 but got {}.", s))),
    }
  }

  fn parse_usize(&mut self) -> Result<usize, ParseError> {
    match self.next() {
      None => self.error("Expected usize but got end of input.".to_owned()),
      Some(s) => s
        .parse()
        .map_err(|_| self.error_raw(format!("Expected usize but got {}.", s))),
    }
  }

  fn parse_tag(&mut self, tag: &str) -> Result<(), ParseError> {
    match self.next() {
      None => self.error(format!("Expected `{}` but got end of input.", tag)),
      Some(s) if s != tag => self.error(format!("Expected `{}` but got {}.", tag, s)),
      _ => Ok(()),
    }
  }

  fn parse_color(&mut self) -> Result<Color, ParseError> {
    let r = self.parse_f64()?;
    let g = self.parse_f64()?;
    let b = self.parse_f64()?;

    Ok(Color { r, g, b })
  }

  fn parse_specular_coefficeint(&mut self) -> Result<f64, ParseError> {
    self.parse_tag("Ns")?;
    self.parse_f64()
  }

  fn parse_ambient_color(&mut self) -> Result<Color, ParseError> {
    self.parse_tag("Ka")?;
    self.parse_color()
  }

  fn parse_diffuse_color(&mut self) -> Result<Color, ParseError> {
    self.parse_tag("Kd")?;
    self.parse_color()
  }

  fn parse_specular_color(&mut self) -> Result<Color, ParseError> {
    self.parse_tag("Ks")?;
    self.parse_color()
  }

  fn parse_emissive_color(&mut self) -> Result<Option<Color>, ParseError> {
    if self.peek() != Some("Ke") {
      return Ok(None);
    }
    self.parse_tag("Ke")?;
    self.parse_color().map(|c| Some(c))
  }

  fn parse_optical_density(&mut self) -> Result<Option<f64>, ParseError> {
    match self.peek() {
      Some("Ni") => {}
      _ => return Ok(None),
    }

    self.parse_tag("Ni")?;
    let optical_density = self.parse_f64()?;
    Ok(Some(optical_density))
  }

  fn parse_dissolve(&mut self) -> Result<f64, ParseError> {
    self.parse_tag("d")?;
    self.parse_f64()
  }

  fn parse_illumination(&mut self) -> Result<Illumination, ParseError> {
    self.parse_tag("illum")?;
    match self.parse_usize()? {
      0 => Ok(Illumination::Ambient),
      1 => Ok(Illumination::AmbientDiffuse),
      2 => Ok(Illumination::AmbientDiffuseSpecular),
      3 => Ok(Illumination::ReflectionRayTrace),
      n => self.error(format!("Unknown illumination model: {}.", n)),
    }
  }

  fn parse_map(&mut self, name: &'static str) -> Result<Option<&'a str>, ParseError> {
    match self.peek() {
      Some(s) => {
        if s != name {
          return Ok(None);
        }
      }
      _ => return Ok(None),
    }

    self.parse_tag(name)?;
    match self.next() {
      None => self.error("Expected texture path but got end of input.".to_owned()),
      Some(s) => Ok(Some(s)),
    }
  }

  fn parse_material(&mut self) -> Result<Material, ParseError> {
    let name = self.parse_newmtl()?;
    self.one_or_more_newlines()?;
    let spec_coeff = self.parse_specular_coefficeint()?;
    self.one_or_more_newlines()?;
    let amb = self.parse_ambient_color()?;
    self.one_or_more_newlines()?;
    let diff = self.parse_diffuse_color()?;
    self.one_or_more_newlines()?;
    let spec = self.parse_specular_color()?;
    self.one_or_more_newlines()?;
    let emit = self.parse_emissive_color()?;
    if emit.is_some() {
      self.one_or_more_newlines()?;
    }
    let optical_density = self.parse_optical_density()?;
    if optical_density.is_some() {
      self.one_or_more_newlines()?;
    }
    let dissolve = self.parse_dissolve()?;
    self.one_or_more_newlines()?;
    let illum = self.parse_illumination()?;
    self.one_or_more_newlines()?;

    // Parse maps
    // Color textures
    let ambient_map = self.parse_map("map_Ka")?;
    if ambient_map.is_some() {
      self.one_or_more_newlines()?;
    }
    let diffuse_map = self.parse_map("map_Kd")?;
    if diffuse_map.is_some() {
      self.one_or_more_newlines()?;
    }
    let specular_map = self.parse_map("map_Ks")?;
    if specular_map.is_some() {
      self.one_or_more_newlines()?;
    }

    // Scalar textures
    let spec_exp_map = self.parse_map("map_Ns")?;
    if spec_exp_map.is_some() {
      self.one_or_more_newlines()?;
    }
    let dissolve_map = self.parse_map("map_d")?;
    if dissolve_map.is_some() {
      self.one_or_more_newlines()?;
    }
    let disp_map = self.parse_map("disp")?;
    if disp_map.is_some() {
      self.one_or_more_newlines()?;
    }

    // Decal (roughness) texture
    let decal = self.parse_map("decal")?;
    if decal.is_some() {
      self.one_or_more_newlines()?;
    }

    // Bump texture
    let mut bump = self.parse_map("map_bump")?;
    if bump.is_none() {
      // Some implementations use map_bump instead
      bump = self.parse_map("map_bump")?;
    }
    if bump.is_some() {
      self.one_or_more_newlines()?;
    }

    Ok(Material {
      name: name.to_owned(),
      specular_coefficient: spec_coeff,
      color_ambient: amb,
      color_diffuse: diff,
      color_specular: spec,
      color_emissive: emit,
      optical_density,
      alpha: dissolve,
      illumination: illum,
      ambient_map: ambient_map.map(|s| s.to_owned()),
      diffuse_map: diffuse_map.map(|s| s.to_owned()),
      specular_map: specular_map.map(|s| s.to_owned()),
      specular_exponent_map: spec_exp_map.map(|s| s.to_owned()),
      dissolve_map: dissolve_map.map(|s| s.to_owned()),
      displacement_map: disp_map.map(|s| s.to_owned()),
      decal_map: decal.map(|s| s.to_owned()),
      bump_map: bump.map(|s| s.to_owned()),
    })
  }

  fn parse_mtlset(&mut self) -> Result<MtlSet, ParseError> {
    self.zero_or_more_newlines();

    let mut ret = Vec::new();

    loop {
      match self.peek() {
        Some("newmtl") => {
          ret.push(self.parse_material()?);
        }
        _ => break,
      }
    }

    match self.peek() {
      None => {}
      Some(s) => return self.error(format!("Expected end of input but got {}.", s)),
    }

    Ok(MtlSet { materials: ret })
  }
}

/// Parses a wavefront `.mtl` file, returning either the successfully parsed
/// file, or an error. Support in this parser for the full file format is
/// best-effort and realistically I will only end up supporting the subset
/// of the file format which falls under the "shit I see exported from blender"
/// category.
pub fn parse<S: AsRef<str>>(input: S) -> Result<MtlSet, ParseError> {
  Parser::new(input.as_ref()).parse_mtlset()
}

#[test]
fn test_parse() {
  use self::Illumination::AmbientDiffuseSpecular;

  let test_case = r#"
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
# emissive color (weighted)
Ke 0.100000 0.100000 0.100000
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
illum 2"#;

  let expected = Ok(MtlSet {
    materials: vec![
      Material {
        name: "Material".to_owned(),
        specular_coefficient: 96.078431,
        color_ambient: Color {
          r: 0.0,
          g: 0.0,
          b: 0.0,
        },
        color_diffuse: Color {
          r: 0.64,
          g: 0.64,
          b: 0.64,
        },
        color_specular: Color {
          r: 0.5,
          g: 0.5,
          b: 0.5,
        },
        color_emissive: Some(Color {
          r: 0.1,
          g: 0.1,
          b: 0.1,
        }),
        optical_density: Some(1.0),
        alpha: 1.0,
        illumination: AmbientDiffuseSpecular,
        ambient_map: None,
        diffuse_map: None,
        specular_map: None,
        specular_exponent_map: None,
        dissolve_map: None,
        displacement_map: None,
        decal_map: None,
        bump_map: None,
      },
      Material {
        name: "None".to_owned(),
        specular_coefficient: 0.0,
        color_ambient: Color {
          r: 0.0,
          g: 0.0,
          b: 0.0,
        },
        color_diffuse: Color {
          r: 0.8,
          g: 0.8,
          b: 0.8,
        },
        color_specular: Color {
          r: 0.8,
          g: 0.8,
          b: 0.8,
        },
        color_emissive: None,
        optical_density: None,
        alpha: 1.0,
        illumination: AmbientDiffuseSpecular,
        ambient_map: None,
        diffuse_map: None,
        specular_map: None,
        specular_exponent_map: None,
        dissolve_map: None,
        displacement_map: None,
        decal_map: None,
        bump_map: None,
      },
    ],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_cube() {
  use self::Illumination::AmbientDiffuseSpecular;

  let test_case = r#"
# Blender MTL File: 'cube.blend'
# Material Count: 1

newmtl Material
Ns 96.078431
Ka 0.000000 0.000000 0.000000
Kd 0.640000 0.640000 0.640000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.000000
d 1.000000
illum 2
map_Ka cube-ambient.png
map_Kd cube-diff.png
map_Ks cube-spec-color.png
map_Ns cube-spec-exp.mps
map_d cube-alpha.mps
disp cube-disp.mps
decal cube-roughness.mps
map_bump cube-bump.mpb
"#;

  let expected = Ok(MtlSet {
    materials: vec![Material {
      name: "Material".to_owned(),
      specular_coefficient: 96.078431,
      color_ambient: Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
      },
      color_diffuse: Color {
        r: 0.64,
        g: 0.64,
        b: 0.64,
      },
      color_specular: Color {
        r: 0.5,
        g: 0.5,
        b: 0.5,
      },
      color_emissive: Some(Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
      }),
      optical_density: Some(1.0),
      alpha: 1.0,
      illumination: AmbientDiffuseSpecular,
      ambient_map: Some("cube-ambient.png".to_owned()),
      diffuse_map: Some("cube-diff.png".to_owned()),
      specular_map: Some("cube-spec-color.png".to_owned()),
      specular_exponent_map: Some("cube-spec-exp.mps".to_owned()),
      dissolve_map: Some("cube-alpha.mps".to_owned()),
      displacement_map: Some("cube-disp.mps".to_owned()),
      decal_map: Some("cube-roughness.mps".to_owned()),
      bump_map: Some("cube-bump.mpb".to_owned()),
    }],
  });

  assert_eq!(parse(test_case), expected);
}

#[test]
fn test_missing_maps() {
  use self::Illumination::AmbientDiffuseSpecular;

  let test_case = r#"
newmtl Material
Ns 96.078431
Ka 0.000000 0.000000 0.000000
Kd 0.640000 0.640000 0.640000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.000000
d 1.000000
illum 2
map_Ka ambient.png
map_Ns spec-exp.mps
map_d alpha.mps
map_bump bump.mpb
"#;

  let expected = Ok(MtlSet {
    materials: vec![Material {
      name: "Material".to_owned(),
      specular_coefficient: 96.078431,
      color_ambient: Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
      },
      color_diffuse: Color {
        r: 0.64,
        g: 0.64,
        b: 0.64,
      },
      color_specular: Color {
        r: 0.5,
        g: 0.5,
        b: 0.5,
      },
      color_emissive: Some(Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
      }),
      optical_density: Some(1.0),
      alpha: 1.0,
      illumination: AmbientDiffuseSpecular,
      ambient_map: Some("ambient.png".to_owned()),
      diffuse_map: None,
      specular_map: None,
      specular_exponent_map: Some("spec-exp.mps".to_owned()),
      dissolve_map: Some("alpha.mps".to_owned()),
      displacement_map: None,
      decal_map: None,
      bump_map: Some("bump.mpb".to_owned()),
    }],
  });

  assert_eq!(parse(test_case), expected);
}
