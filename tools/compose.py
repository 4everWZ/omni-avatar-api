"""Run a supported Compose stack with repository-relative paths on every OS."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / 'deploy' / 'compose'
TARGETS = {
    'omni': ('omni-node',),
    'gateway': ('gateway-node',),
    'avatar': ('avatar-node',),
    'images': ('image-node',),
    'llm': ('llm', 'llm-endpoint'),
    'workstation': ('base', 'workstation', 'avatar', 'llm', 'inference'),
    'workstation-omni': ('base', 'workstation', 'avatar'),
    'workstation-bf16': ('base', 'workstation', 'workstation-bf16', 'avatar'),
    'orin': ('base',),
    'orin-cu129': ('base', 'orin-cu129'),
}


def command(target, compose_args, *, images=False, remote_images=False, vpn=False, media=False,
            registered_avatars=False, env_file=None):
    if target not in TARGETS:
        raise ValueError(f'Unknown deployment target: {target}')
    if images and target != 'workstation':
        raise ValueError('The images overlay requires the split workstation target')
    if images and remote_images:
        raise ValueError('Choose either shared or dedicated image placement')
    if (remote_images or vpn or media) and target in ('images', 'llm', 'avatar', 'omni'):
        raise ValueError('Gateway overlays require a deployment target containing the gateway')
    if media and target == 'gateway':
        raise ValueError('The program media overlay requires the shared stack network')
    files = list(TARGETS[target])
    if registered_avatars:
        if not {'avatar', 'avatar-node'}.intersection(files):
            raise ValueError('Registered avatars require a target running the Avatar service')
        files.append('registered-avatars')
    files.extend(name for name, enabled in [('images', images), ('remote-images', remote_images),
                                          ('vpn', vpn), ('media', media)] if enabled)
    env_file = Path(env_file).resolve() if env_file else ROOT / '.env'
    args = ['docker', 'compose', '--project-directory', str(ROOT), '--env-file', str(env_file)]
    for name in files:
        args.extend(['-f', str(COMPOSE / f'{name}.yml')])
    return args + list(compose_args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Put Docker Compose arguments after --. Example: '
        'python tools/compose.py --vpn -- up -d --wait'))
    parser.add_argument('--target', choices=TARGETS, default='workstation')
    parser.add_argument('--registered-avatars', action='store_true', help='Mount a workstation-owned Avatar registry')
    parser.add_argument('--images', action='store_true', help='Enable queued Klein image generation')
    parser.add_argument('--remote-images', action='store_true', help='Route image jobs to a dedicated worker')
    parser.add_argument('--vpn', action='store_true', help='Publish on OMNI_GATEWAY_VPN_BIND as well as loopback')
    parser.add_argument('--media', action='store_true', help='Enable optional MediaMTX transport')
    parser.add_argument('--env-file', type=Path, help='Defaults to the repository .env, even from another directory')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without contacting Docker')
    options = parser.parse_args(argv[:split])
    if split == len(argv) or not argv[split + 1:]:
        parser.error('Specify Docker Compose arguments after --')
    try:
        args = command(options.target, argv[split + 1:],
                       images=options.images, remote_images=options.remote_images, vpn=options.vpn, media=options.media,
                       registered_avatars=options.registered_avatars, env_file=options.env_file)
    except ValueError as error:
        parser.error(str(error))
    if options.dry_run:
        print(shlex.join(args))
        return 0
    try:
        return subprocess.call(args, cwd=ROOT)
    except FileNotFoundError:
        parser.exit(127, 'Docker was not found. Install Docker with the Compose plugin.\n')


if __name__ == '__main__':
    raise SystemExit(main())
