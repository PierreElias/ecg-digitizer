#!/usr/bin/env python3
"""
ECG Digitizer Test Runner

Run this script to execute all unit tests for waveform extraction
and layout identification, and generate a detailed report.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --waveform   # Run only waveform tests
    python run_tests.py --layout     # Run only layout tests
    python run_tests.py --report     # Generate JSON report
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Open-ECG-Digitizer'))

from tests.test_waveform_extraction import WaveformExtractionTestSuite
from tests.test_layout_identification import LayoutIdentificationTestSuite


def print_banner():
    print()
    print("=" * 70)
    print("   ECG DIGITIZER TEST SUITE")
    print("   Testing Waveform Extraction and Layout Identification")
    print("=" * 70)
    print()


def print_result(result, verbose=True):
    """Print a single test result."""
    status = "✅ PASS" if result['passed'] else "❌ FAIL"
    print(f"  {status}: {result['name']}")
    if verbose:
        print(f"         {result['message']}")
        if result['details'] and not result['passed']:
            print(f"         Details: {json.dumps(result['details'], indent=2)[:200]}...")


def run_waveform_tests(verbose=True):
    """Run waveform extraction tests."""
    print("\n" + "-" * 50)
    print("WAVEFORM EXTRACTION TESTS")
    print("-" * 50)

    suite = WaveformExtractionTestSuite()
    results = suite.run_all_tests()

    for result in results:
        print_result(result, verbose)

    passed = sum(1 for r in results if r['passed'])
    print(f"\n  Summary: {passed}/{len(results)} tests passed")

    return results


def run_layout_tests(verbose=True):
    """Run layout identification tests."""
    print("\n" + "-" * 50)
    print("LAYOUT IDENTIFICATION TESTS")
    print("-" * 50)

    suite = LayoutIdentificationTestSuite()
    results = suite.run_all_tests()

    for result in results:
        print_result(result, verbose)

    passed = sum(1 for r in results if r['passed'])
    print(f"\n  Summary: {passed}/{len(results)} tests passed")

    return results


def generate_report(waveform_results, layout_results):
    """Generate a detailed JSON report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': len(waveform_results) + len(layout_results),
            'passed': sum(1 for r in waveform_results + layout_results if r['passed']),
            'failed': sum(1 for r in waveform_results + layout_results if not r['passed']),
            'pass_rate': round(
                sum(1 for r in waveform_results + layout_results if r['passed']) /
                max(len(waveform_results) + len(layout_results), 1) * 100, 1
            )
        },
        'waveform_extraction': {
            'results': waveform_results,
            'passed': sum(1 for r in waveform_results if r['passed']),
            'total': len(waveform_results)
        },
        'layout_identification': {
            'results': layout_results,
            'passed': sum(1 for r in layout_results if r['passed']),
            'total': len(layout_results)
        }
    }

    # Save report
    report_dir = Path(__file__).parent / 'test_reports'
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f'test_report_{timestamp}.json'

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description='ECG Digitizer Test Runner')
    parser.add_argument('--waveform', action='store_true', help='Run only waveform tests')
    parser.add_argument('--layout', action='store_true', help='Run only layout tests')
    parser.add_argument('--report', action='store_true', help='Generate JSON report')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')

    args = parser.parse_args()

    print_banner()

    verbose = not args.quiet
    waveform_results = []
    layout_results = []

    # Run tests based on arguments
    if args.waveform and not args.layout:
        waveform_results = run_waveform_tests(verbose)
    elif args.layout and not args.waveform:
        layout_results = run_layout_tests(verbose)
    else:
        # Run all tests
        waveform_results = run_waveform_tests(verbose)
        layout_results = run_layout_tests(verbose)

    # Print overall summary
    all_results = waveform_results + layout_results
    total = len(all_results)
    passed = sum(1 for r in all_results if r['passed'])
    failed = total - passed

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Total Tests:  {total}")
    print(f"  Passed:       {passed} ({round(passed/total*100) if total else 0}%)")
    print(f"  Failed:       {failed} ({round(failed/total*100) if total else 0}%)")
    print("=" * 70)

    # Generate report if requested
    if args.report:
        generate_report(waveform_results, layout_results)

    # Return exit code based on test results
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
