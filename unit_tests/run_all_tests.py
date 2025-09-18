#!/usr/bin/env python3
"""
Automated test runner for AirEvolve2 unit tests.

This script provides comprehensive test execution with:
- Category-based test running
- Visual mode support
- Progress reporting
- Summary statistics
- Failure analysis
"""

import argparse
import sys
import os
import time
import unittest
import subprocess
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import importlib.util


class TestCategory:
    """Test category configuration."""
    
    def __init__(self, name: str, description: str, test_files: List[str], 
                 supports_visual: bool = False, requires_deps: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.test_files = test_files
        self.supports_visual = supports_visual
        self.requires_deps = requires_deps or []


class TestRunner:
    """Main test runner class."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.categories = self._define_categories()
        
    def _define_categories(self) -> Dict[str, TestCategory]:
        """Define test categories and their configurations."""
        return {
            'core': TestCategory(
                name='Core Functionality',
                description='Basic genome handler operations and genetic algorithms',
                test_files=[
                    'test_cartesian_euler_core.py',
                    'test_cartesian_euler_integration.py',
                    'test_spherical_angular_genome_handler.py',
                    'test_operator_integration.py'
                ]
            ),
            'symmetry': TestCategory(
                name='Symmetry Operations',
                description='Bilateral and spherical symmetry functionality',
                test_files=[
                    'test_bilateral_symmetry.py',
                    'test_spherical_symmetry.py'
                ]
            ),
            'repair': TestCategory(
                name='Particle Repair',
                description='Collision detection, repair performance, and symmetry preservation (supports Cartesian and Spherical)',
                test_files=[
                    'test_particle_repair.py'
                ]
            ),
            'statistical': TestCategory(
                name='Statistical Analysis',
                description='Statistical uniformity and validation tests',
                test_files=[
                    'test_statistical_uniformity.py'
                ],
                requires_deps=['scipy']
            ),
            'visualization': TestCategory(
                name='Visualization',
                description='Visual inspection and plotting capabilities',
                test_files=[
                    'test_visualization.py'
                ],
                supports_visual=True,
                requires_deps=['matplotlib']
            ),
        }
    
    def check_dependencies(self, category: TestCategory) -> Tuple[bool, List[str]]:
        """Check if required dependencies are available."""
        missing_deps = []
        
        for dep in category.requires_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        return len(missing_deps) == 0, missing_deps
    
    def get_available_test_files(self) -> List[str]:
        """Get list of all available test files."""
        test_files = []
        for test_file in self.test_dir.glob('test_*.py'):
            if test_file.name != 'test_template.py':  # Exclude template
                test_files.append(test_file.name)
        return sorted(test_files)
    
    def run_test_file(self, test_file: str, visual: bool = False, verbose: bool = False, 
                      genome_handler: str = None) -> Tuple[bool, str, float]:
        """Run a single test file and return results."""
        test_path = self.test_dir / test_file
        
        if not test_path.exists():
            return False, f"Test file not found: {test_file}", 0.0
        
        # Prepare command
        cmd = [sys.executable, str(test_path)]
        if visual and 'visualization' in test_file:
            cmd.append('--visual')
        if verbose:
            cmd.append('-v')
        if genome_handler and 'particle_repair' in test_file:
            cmd.extend(['--genome-handler', genome_handler])
        
        # Run test
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.test_dir.parent  # Run from project root
            )
            end_time = time.time()
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            duration = end_time - start_time
            
            return success, output, duration
            
        except subprocess.TimeoutExpired:
            return False, f"Test timed out after 5 minutes", time.time() - start_time
        except Exception as e:
            return False, f"Error running test: {str(e)}", time.time() - start_time
    
    def run_category(self, category_name: str, visual: bool = False, verbose: bool = False,
                     genome_handler: str = None) -> Dict:
        """Run all tests in a category."""
        if category_name not in self.categories:
            return {'success': False, 'error': f'Unknown category: {category_name}'}
        
        category = self.categories[category_name]
        
        # Check dependencies
        deps_ok, missing_deps = self.check_dependencies(category)
        if not deps_ok:
            return {
                'success': False, 
                'error': f'Missing dependencies: {", ".join(missing_deps)}',
                'skipped': True
            }
        
        print(f"\n{'='*60}")
        print(f"Running {category.name} Tests")
        print(f"{'='*60}")
        print(f"Description: {category.description}")
        print(f"Test files: {len(category.test_files)}")
        
        results = {
            'category': category.name,
            'total_files': len(category.test_files),
            'passed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_time': 0.0,
            'file_results': [],
            'success': True
        }
        
        for test_file in category.test_files:
            print(f"\n📝 Running {test_file}...")
            
            # Check if file exists
            if not (self.test_dir / test_file).exists():
                print(f"   ⚠️  File not found, skipping")
                results['skipped_files'] += 1
                results['file_results'].append({
                    'file': test_file,
                    'status': 'skipped',
                    'reason': 'File not found',
                    'duration': 0.0
                })
                continue
            
            success, output, duration = self.run_test_file(test_file, visual, verbose, genome_handler)
            results['total_time'] += duration
            
            if success:
                print(f"   ✅ Passed ({duration:.2f}s)")
                results['passed_files'] += 1
                status = 'passed'
            else:
                print(f"   ❌ Failed ({duration:.2f}s)")
                results['failed_files'] += 1
                results['success'] = False
                status = 'failed'
                
                # Show first few lines of error for immediate feedback
                if output:
                    error_lines = output.split('\n')[-10:]  # Last 10 lines
                    print(f"   Error preview:")
                    for line in error_lines:
                        if line.strip():
                            print(f"      {line}")
            
            results['file_results'].append({
                'file': test_file,
                'status': status,
                'duration': duration,
                'output': output
            })
        
        return results
    
    def run_all_categories(self, visual: bool = False, verbose: bool = False, 
                          exclude_categories: Optional[List[str]] = None,
                          genome_handler: str = None) -> Dict:
        """Run all test categories."""
        exclude_categories = exclude_categories or []
        
        print("🚀 Starting AirEvolve2 Unit Test Suite")
        print(f"Test directory: {self.test_dir}")
        print(f"Visual mode: {'Enabled' if visual else 'Disabled'}")
        print(f"Verbose mode: {'Enabled' if verbose else 'Disabled'}")
        
        if exclude_categories:
            print(f"Excluding categories: {', '.join(exclude_categories)}")
        
        overall_start = time.time()
        all_results = {
            'total_categories': 0,
            'passed_categories': 0,
            'failed_categories': 0,
            'skipped_categories': 0,
            'total_files': 0,
            'passed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_time': 0.0,
            'category_results': [],
            'success': True
        }
        
        for category_name, category in self.categories.items():
            if category_name in exclude_categories:
                print(f"\n⏭️  Skipping {category.name} (excluded)")
                all_results['skipped_categories'] += 1
                continue
            
            # Skip visualization if visual mode not supported and not requested
            if not visual and category.supports_visual and 'visualization' in category_name:
                print(f"\n⏭️  Skipping {category.name} (requires --visual)")
                all_results['skipped_categories'] += 1
                continue
            
            all_results['total_categories'] += 1
            result = self.run_category(category_name, visual, verbose, genome_handler)
            
            if result.get('skipped', False):
                all_results['skipped_categories'] += 1
                print(f"\n⚠️  Skipped {category.name}: {result.get('error', 'Unknown reason')}")
            elif result['success']:
                all_results['passed_categories'] += 1
                print(f"\n✅ {category.name} completed successfully")
            else:
                all_results['failed_categories'] += 1
                all_results['success'] = False
                print(f"\n❌ {category.name} failed")
            
            # Aggregate file statistics
            all_results['total_files'] += result.get('total_files', 0)
            all_results['passed_files'] += result.get('passed_files', 0)
            all_results['failed_files'] += result.get('failed_files', 0)
            all_results['skipped_files'] += result.get('skipped_files', 0)
            all_results['total_time'] += result.get('total_time', 0.0)
            
            all_results['category_results'].append(result)
        
        overall_end = time.time()
        all_results['total_time'] = overall_end - overall_start
        
        return all_results
    
    def print_summary(self, results: Dict):
        """Print comprehensive test summary."""
        print(f"\n{'='*80}")
        print("🏁 TEST SUMMARY")
        print(f"{'='*80}")
        
        # Overall status
        if results['success']:
            print("🎉 Overall Status: PASSED")
        else:
            print("💥 Overall Status: FAILED")
        
        print(f"\n📊 Statistics:")
        print(f"   Categories: {results['passed_categories']}/{results['total_categories']} passed")
        print(f"   Test Files: {results['passed_files']}/{results['total_files']} passed")
        print(f"   Total Time: {results['total_time']:.2f} seconds")
        
        if results['failed_categories'] > 0:
            print(f"   Failed Categories: {results['failed_categories']}")
        if results['skipped_categories'] > 0:
            print(f"   Skipped Categories: {results['skipped_categories']}")
        if results['failed_files'] > 0:
            print(f"   Failed Files: {results['failed_files']}")
        if results['skipped_files'] > 0:
            print(f"   Skipped Files: {results['skipped_files']}")
        
        # Category breakdown
        print(f"\n📋 Category Results:")
        for result in results['category_results']:
            category = result.get('category', 'Unknown')
            if result.get('skipped', False):
                status = '⏭️  SKIPPED'
                reason = result.get('error', 'Unknown reason')
                print(f"   {status:<12} {category:<20} ({reason})")
            elif result['success']:
                status = '✅ PASSED'
                time_str = f"{result['total_time']:.2f}s"
                files_str = f"{result['passed_files']}/{result['total_files']} files"
                print(f"   {status:<12} {category:<20} ({files_str}, {time_str})")
            else:
                status = '❌ FAILED'
                time_str = f"{result['total_time']:.2f}s"
                files_str = f"{result['passed_files']}/{result['total_files']} files"
                print(f"   {status:<12} {category:<20} ({files_str}, {time_str})")
        
        # Failed files details
        if results['failed_files'] > 0:
            print(f"\n❌ Failed Files Details:")
            for result in results['category_results']:
                for file_result in result.get('file_results', []):
                    if file_result['status'] == 'failed':
                        print(f"   📄 {file_result['file']}")
                        if 'output' in file_result and file_result['output']:
                            # Show last few lines of error
                            error_lines = file_result['output'].split('\n')[-5:]
                            for line in error_lines:
                                if line.strip():
                                    print(f"      {line}")
                        print()
        
        # Performance insights
        if results['total_files'] > 0:
            avg_time = results['total_time'] / results['total_files']
            print(f"\n⚡ Performance:")
            print(f"   Average time per file: {avg_time:.2f}s")
            
            # Show slowest files
            slow_files = []
            for result in results['category_results']:
                for file_result in result.get('file_results', []):
                    if file_result['status'] in ['passed', 'failed']:
                        slow_files.append((file_result['file'], file_result['duration']))
            
            slow_files.sort(key=lambda x: x[1], reverse=True)
            if slow_files:
                print(f"   Slowest files:")
                for file_name, duration in slow_files[:3]:  # Top 3
                    print(f"      {file_name}: {duration:.2f}s")
    
    def list_categories(self):
        """List all available test categories."""
        print("📚 Available Test Categories:")
        print()
        
        for category_name, category in self.categories.items():
            deps_ok, missing_deps = self.check_dependencies(category)
            
            print(f"🏷️  {category.name} ({category_name})")
            print(f"   Description: {category.description}")
            print(f"   Files: {len(category.test_files)}")
            print(f"   Visual support: {'Yes' if category.supports_visual else 'No'}")
            
            if category.requires_deps:
                if deps_ok:
                    print(f"   Dependencies: {', '.join(category.requires_deps)} ✅")
                else:
                    print(f"   Dependencies: {', '.join(category.requires_deps)} ❌ (missing: {', '.join(missing_deps)})")
            
            print(f"   Test files:")
            for test_file in category.test_files:
                exists = (self.test_dir / test_file).exists()
                status = "✅" if exists else "❌"
                print(f"      {status} {test_file}")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run AirEvolve2 unit tests with comprehensive reporting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py                                    # Run all tests
  python run_all_tests.py --visual                           # Run all tests with visualization
  python run_all_tests.py --category repair                  # Run repair tests (Cartesian)
  python run_all_tests.py --category repair --genome-handler spherical  # Run repair tests (Spherical)
  python run_all_tests.py --genome-handler spherical         # Run all tests with spherical handler where applicable
  python run_all_tests.py --list                             # List available categories
  python run_all_tests.py --exclude repair                   # Run all except repair tests
        """
    )
    
    parser.add_argument(
        '--category', '-c',
        help='Run specific test category (use --list to see available categories)'
    )
    
    parser.add_argument(
        '--visual', '-v',
        action='store_true',
        help='Enable visual mode for visualization tests'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose test output'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available test categories and exit'
    )
    
    parser.add_argument(
        '--exclude', '-e',
        action='append',
        help='Exclude specific categories (can be used multiple times)'
    )
    
    parser.add_argument(
        '--file', '-f',
        help='Run a specific test file'
    )
    
    parser.add_argument(
        '--genome-handler',
        choices=['cartesian', 'spherical'],
        help='Genome handler type for particle repair tests (cartesian or spherical)'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.list:
        runner.list_categories()
        return 0
    
    if args.file:
        # Run single file
        print(f"🧪 Running single test file: {args.file}")
        success, output, duration = runner.run_test_file(args.file, args.visual, args.verbose, args.genome_handler)
        
        if success:
            print(f"✅ {args.file} passed ({duration:.2f}s)")
            if args.verbose and output:
                print(output)
            return 0
        else:
            print(f"❌ {args.file} failed ({duration:.2f}s)")
            print(output)
            return 1
    
    if args.category:
        # Run specific category
        results = runner.run_category(args.category, args.visual, args.verbose, args.genome_handler)
        
        if results.get('skipped', False):
            print(f"\n⚠️  Category skipped: {results.get('error', 'Unknown reason')}")
            return 2
        
        success = results['success']
        
        # Print mini summary for single category
        print(f"\n{'='*50}")
        print(f"Category: {results['category']}")
        print(f"Files: {results['passed_files']}/{results['total_files']} passed")
        print(f"Time: {results['total_time']:.2f}s")
        print(f"Status: {'PASSED' if success else 'FAILED'}")
        
        return 0 if success else 1
    
    # Run all categories
    results = runner.run_all_categories(args.visual, args.verbose, args.exclude, args.genome_handler)
    runner.print_summary(results)
    
    return 0 if results['success'] else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)