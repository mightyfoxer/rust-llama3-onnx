use ort::{inputs, GraphOptimizationLevel, Session};
// use std::{env, process};
use ndarray::{array, concatenate, s, Array1, ArrayViewD, Axis};
use rand::Rng;
use std::{
    io::{self, Write},
    path::Path,
};
use tokenizers::Tokenizer;
const PROMPT: &str =
    "The corsac fox (Vulpes corsac), also known simply as a corsac, is a medium-sized fox found in";
/// Max tokens to generate
const GEN_TOKENS: i32 = 90;
/// Top_K -> Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.
const TOP_K: usize = 5;

fn init_runtime() -> ort::Result<()> {
    ort::init()
        .with_name("llama")
        .with_execution_providers([ort::CUDAExecutionProvider::default().build()])
        .commit()?;
    Ok(())
}

fn load_model(model_path: &str) -> ort::Result<Session> {
    println!("Loading model...");
    let session_builder = Session::builder();
    match session_builder {
        Ok(builder) => {
            let model = builder
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(model_path);
            match model {
                Ok(model) => {
                    println!("Model loaded successfully");
                    Ok(model)
                }
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    }
}

fn tokenize_prompt(prompt: &str, tokenizer: &Tokenizer) -> io::Result<Array1<i64>> {
    match tokenizer.encode(prompt, false) {
        Ok(tokens) => {
            let token_ids = tokens
                .get_ids()
                .iter()
                .map(|i| *i as i64)
                .collect::<Vec<i64>>();
            Ok(Array1::from_vec(token_ids))
        }
        Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
    }
}

fn generate_text(
    session: &Session,
    tokenizer: &Tokenizer,
    mut tokens: Array1<i64>,
) -> ort::Result<String> {
    let mut stdout = io::stdout();
    let mut rng = rand::thread_rng();
    let mut generated_text = String::new();

    for _ in 0..GEN_TOKENS {
        let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
        let outputs = session.run(inputs![array]?)?;
        let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_tensor()?;

        // Collect and sort logits
        let mut probabilities = generated_tokens
            .slice(s![0, 0, -1, ..])
            .to_owned()
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<_>>();
        probabilities
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        // Sample using top-k sampling
        let token = probabilities[rng.gen_range(0..=TOP_K.min(probabilities.len() - 1))].0;
        tokens = concatenate![Axis(0), tokens, array![token.try_into().unwrap()]];

        let token_str = tokenizer.decode(&[token as _], true).unwrap();
        generated_text.push_str(&token_str);
        print!("{}", token_str);
        stdout.flush().unwrap();
    }

    Ok(generated_text)
}

fn main() -> ort::Result<()> {
    let mut stdout: io::Stdout = io::stdout();

    match init_runtime() {
        Ok(_) => println!("Runtime initialized."),
        Err(e) => return Err(e),
    }

    let session = match load_model("Your Model Path") {
        Ok(mdl) => mdl,
        Err(err) => return Err(err),
    };

    let tokenizer =
        match tokenizers::Tokenizer::from_file(Path::new("Your Tokenizer.json file path")) {
            Ok(tokenizer) => tokenizer,
            Err(err) => return Err(ort::Error::CustomError(err)),
        };
    stdout.flush().unwrap();
    let tokens = match tokenize_prompt(PROMPT, &tokenizer) {
        Ok(tokens) => tokens,
        Err(e) => {
            let boxed_error: Box<dyn std::error::Error + Send + Sync> = Box::new(e);
            return Err(ort::Error::from(boxed_error));
        }
    };
    println!("{}", tokens);
    let generated_text = match generate_text(&session, &tokenizer, tokens) {
        Ok(text) => text,
        Err(e) => return Err(e),
    };
    println!("\nGenerated text:\n{}", generated_text);
    Ok(())
}
