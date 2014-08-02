//! The entry point, bitch.

#![crate_type = "lib"]
#![deny(warnings)]
#![deny(missing_doc)]
#![allow(dead_code)]

pub use lex::ParseError;

mod lex;

pub mod mtl;
pub mod obj;
