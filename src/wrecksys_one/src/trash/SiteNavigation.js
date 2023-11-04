'use client'

import { AppBar, Box, ButtonGroup, IconButton, Tab, Tabs, Toolbar } from '@mui/material'
import Image from 'next/image'
import Button from '@mui/material/Button'
import Container from '@mui/material/Container'
import { useState } from 'react'
import Grid from '@mui/material/Unstable_Grid2'

function a11yProps (index) {
  return {
    id: `tab-${index}`,
    'aria-controls': `navpanel-${index}`
  }
}

export default function SiteNavigation () {
  const [location, setLocation] = useState(0)
  const navigateTo = (event, newValue) => {
    setLocation(newValue)
  }

  return (
    <AppBar position='fixed' sx={{ minHeight: 40 }} enableColorOnDark>
      <Container maxWidth='md' disableGutters>
        <Toolbar disableGutters>
          <Grid container>
            <Grid key='logo'>
              <Image src='/head2.svg' alt='Monkey Head' height={40} width={40} />
              <Image
                src='/text-logo.svg'
                alt='WreckSys Logo'
                height={30}
                width={120}
                style={{ filter: 'invert(61%) sepia(19%) saturate(267%) hue-rotate(92deg) brightness(91%) contrast(87%)' }}
              />
            </Grid>

            <Grid key='menu'>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs
                  value={location}
                  onChange={navigateTo}
                  textColor='inherit'
                  indicatorColor='secondary'
                  aria-label='navpanel'
                  sx={{ position: 'absolute', bottom: 0, p: 0, m: 0 }}
                >
                  <Tab color='inherit' label='Home' {...a11yProps(0)} />
                  <Tab color='inherit' label='Data' {...a11yProps(1)} />
                  <Tab color='inherit' label='App' {...a11yProps(2)} />
                  <Tab color='inherit' label='Code' {...a11yProps(3)} />
                </Tabs>
              </Box>
            </Grid>
          </Grid>
        </Toolbar>
      </Container>
    </AppBar>
  )
}
