//! Parsers for wavefront's `.obj` and `.mtl` file format for loading meshes.
#![crate_type = "lib"]
#![deny(warnings)]
#![deny(missing_docs)]
#![allow(unstable)]
#![feature(unboxed_closures)]

pub use lex::ParseError;

mod lex;

pub mod mtl;
pub mod obj;
