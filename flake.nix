# flake.nix
#
# The buildable package a NixOS system flake consumes. This file exports `kb`
# — both binaries, `kb` and `kb-mcp` — and an overlay; it exports no NixOS
# module, because a host's services belong in that host's own configuration,
# and a service definition authored here would be a second place the host's
# shape is decided.
#
# The derivation asserts that the committed tree builds. It does not run the
# test suite: the suite spawns the built binaries, reads a home directory and
# reaches embedding and reranking endpoints that a Nix sandbox does not have,
# and `mise run check` is the gate for correctness.
#
#   nix build .#kb            # ./result/bin/{kb,kb-mcp}
#   nix run .#kb -- --help
#   nix run .#kb-mcp -- --help
#   nix flake check

{
  description = "kb — personal knowledge base: content-addressed store, derived index, MCP server";

  inputs = {
    # The same channel the systems flake tracks; that flake makes this input
    # follow its own nixpkgs, so one rustc builds everything on the host.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);

      # Only what the build reads. A change to the Python gate, the eval
      # records or the plans must not rebuild the binary.
      src = nixpkgs.lib.fileset.toSource {
        root = ./.;
        fileset = nixpkgs.lib.fileset.unions [
          ./Cargo.toml
          ./Cargo.lock
          ./README.md
          ./src
          # src/prompt.rs includes the cold-start template at compile time.
          ./templates
        ];
      };

      mkKb =
        pkgs:
        pkgs.rustPlatform.buildRustPackage {
          pname = "kb";
          version = cargoToml.package.version;
          inherit src;

          # No git sources in the lockfile, so the lockfile alone is the
          # dependency closure; nothing to vendor by hand.
          cargoLock.lockFile = ./Cargo.lock;

          # Every dependency is pure Rust — reqwest on rustls, rusqlite
          # bundled, gix with sha1 only — so there is nothing native to link.
          doCheck = false;

          meta = {
            description = cargoToml.package.description;
            homepage = "https://github.com/tftio/kb";
            license = pkgs.lib.licenses.mit;
            mainProgram = "kb";
            platforms = systems;
          };
        };
    in
    {
      packages = forAllSystems (pkgs: rec {
        kb = mkKb pkgs;
        default = kb;
      });

      apps = forAllSystems (pkgs: {
        kb = {
          type = "app";
          program = "${self.packages.${pkgs.system}.kb}/bin/kb";
          meta.description = "kb command-line interface";
        };
        kb-mcp = {
          type = "app";
          program = "${self.packages.${pkgs.system}.kb}/bin/kb-mcp";
          meta.description = "kb MCP server, stdio or --http";
        };
        default = self.apps.${pkgs.system}.kb;
      });

      overlays.default = final: prev: { kb = mkKb final; };

      # `nix flake check` builds the package and asks both binaries for help,
      # which is the cheapest proof that the thing that built also starts.
      checks = forAllSystems (pkgs: {
        build = self.packages.${pkgs.system}.kb;
        binaries-start =
          pkgs.runCommand "kb-binaries-start"
            {
              nativeBuildInputs = [ self.packages.${pkgs.system}.kb ];
            }
            ''
              kb --help > /dev/null
              kb-mcp --help > /dev/null
              touch $out
            '';
      });
    };
}
